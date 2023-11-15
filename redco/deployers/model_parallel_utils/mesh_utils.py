#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
提供了与模型并行相关的工具函数和类，支持对模型参数进行分片和分区
"""

import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict
import optax

from .partition_utils import set_partitions


def get_mesh(n_model_shards):
    """
    创建模型并行的mesh对象,该对象将设备分为两个维度:数据并行维度dp和模型并行维度mp
    n_model_shards:模型并行的分片数
    """
    if n_model_shards == 1:
        return None

    assert jax.device_count() % n_model_shards == 0           #不满足倍数关系的话会引发异常

    mesh_devices = np.array(jax.devices()).reshape(           #将设备根据n_model_shards分为jax.device_count() // n_model_shards行, n_model_shards列的矩阵
        jax.device_count() // n_model_shards, n_model_shards)
    mesh = Mesh(mesh_devices, ('dp', 'mp'))     #mesh_devices 的行数表示数据并行的数量，每行中的每个元素表示一个数据并行维度上的设备。而列数表示模型并行的数量，每列中的每个元素表示一个模型并行维度上的设备

    return mesh


class ShapeDtypeStruct:
    """
    这个类的主要目的是在get_sharding_rules函数中创建一个表示张量形状和数据类型的结构,以便在分析模型参数时能够更轻松地
    确定如何进行分片。这个类的实例在get_sharding_rules函数中用于表示参数的形状和数据类型的信息,方便后续的规则判断
    """
    __slots__ = ["shape", "dtype"]     #用于定义类的属性名

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def get_param_spec(params, params_sharding_rules):
    """
    为模型参数创建分片规则。在分布式训练中，模型参数会被分配到不同的设备上进行计算，因此需要定于参数分配规则
    params:包含模型参数的结构化数据，通常是一个嵌套的字典或类似结构
    params_sharding_rules:一个描述参数分片规则的结构，通常是一个由元组组成的列表
    函数通过对params进行遍历,根据params_sharding_rules中定义的规则为每个参数设置相应的分片规则。最终,返回一个与输入params结构相同的新结构,其中每个参数都被标记为分片到哪个设备上
    """
    return set_partitions(unfreeze(params), params_sharding_rules)


def shard_params(params, params_spec, mesh):
    """
    函数作用是将模型参数按照预定义的规则进行分片。推断时使用
    params:模型参数
    params_spec:分片规则
    mesh:模型的拓扑结构
    """

    #使用pjit定义并行处理函数，该函数接收一个参数x，对应于模型的参数结构
    #in_shardings被设置为None，表示输入的参数不需要分片
    #out_shardings=params_spec，表示输出的参数需要按照给定规则分片
    shard_fn = pjit(
        lambda x: x, in_shardings=(None,), out_shardings=params_spec)

    with mesh:
        #在这个代码块中的计算将在mesh对象定义的设备上执行，例如在数据并行的环境中，不同设备上的数据会被同时处理
        return shard_fn(params)


def shard_params_and_opt_state(params, params_spec, mesh, optimizer):
    """
    用于将模型参数和优化器状态进行分片
    在分布式训练中，由于模型参数和优化器状态需要在多个设备之间共享，因此需要确保他们的结构与模型的并行结构相匹配
    以便能够正确的进行分片和共享，这就是获取优化器状态的规格的目的
    """
    #定义初始化函数，用于初始化优化器状态
    def init_fn(params_):
        opt_state_ = optimizer.init(params_)
        return opt_state_, params_
    #获取优化器状态的规格，此规格指定了优化器状态中的哪些部分需要被分片，以适应模型的并行结构
    def get_opt_spec(x):                                 #x表示模型的参数或者优化器的状态
        if isinstance(x, (dict, FrozenDict,)):           #如果x是两者之一，即为模型参数，返回之前定义的模型参数分片规则
            return params_spec
        return None           #不需要分片

    params_shapes = jax.tree_util.tree_map(                         #将模型参数转换为一个包含每个参数形状和数据类型的shapedtypestruct对象的树形结构
        lambda x: ShapeDtypeStruct(x.shape, x.dtype), params)
    state_shapes = jax.eval_shape(init_fn, params_shapes)           #获取模型初始化后的状态的形状信息

    #获取模型初始化之后的状态的规格，即opt_state_spec，此规格将用于指导如何对优化器状态进行分片，以适应模型的并行结构
    opt_state_spec, _ = jax.tree_util.tree_map(
        get_opt_spec, state_shapes,
        is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,)))

    p_get_initial_state = pjit(
        init_fn,
        in_shardings=(params_spec,),
        out_shardings=(opt_state_spec, params_spec))

    with mesh:
        opt_state, params = p_get_initial_state(params)

    return params, opt_state, opt_state_spec


def gather_params_to_cpu(params, params_spec, mesh):
    """
    将分布式环境中的参数收集到CPU设备上,具体而言,将参数params按照规定的分片规则params_spec进行集中收集
    """
    def param_gather_fn(param_spec):
        return pjit(
            lambda x: x, in_shardings=(param_spec, ), out_shardings=None)       #输入按规则分片，输出不再需要进行分片，即收集到一个设备上

    gather_fns = jax.tree_util.tree_map(                                        #创建一个与params_spec结构相同的树，其中每个叶子节点都是一个函数，用于在收集参数时在设备之间进行转移
        lambda param_spec: param_gather_fn(param_spec),
        params_spec,
        is_leaf=lambda x: x is None or isinstance(x, P))

    #使用jax.tree_util.tree_map遍历gather_fns和params，对每个参数应用相应的收集函数，然后使用jax.device_get将结果从设备上获取到CPU
    with mesh:
        with jax.default_device(jax.devices('cpu')[0]):        
            return jax.tree_util.tree_map(
                lambda gather_fn, param: jax.device_get(gather_fn(param)),
                gather_fns,
                params)


def get_sharding_rules(params, mesh_model_shards, investigate_depth=2):
    """
    根据参数的形状和名称生成参数分片的规则，主要用于确定在模型并行设置中，哪些参数需要在模型并行维度上进行分片
    mesh_model_shards:模型并行的设备数量，即模型在模型并行的维度上被分成的块数
    """
    sharding_rules = {
        ('(bias|scale)',): None,                    #正则表达式，包含偏置和缩放两个字段的参数不进行分片，每个设备上都有完整的偏置项和缩放项的拷贝
        ('embedding',): P('mp', None),              #包含embedding字段的参数需要在模型并行维度上进行分片
    }

    last_dense_mp_dim = None                        #用于追踪上一个被分片的维度，优化策略，避免重复搜索
    flat_params = flatten_dict(params)              #将模型参数扁平化为一个字典的表示，字典的值是参数的路径，值是参数本身，扁平化目的是方便遍历
    for key in sorted(flat_params.keys(), key=lambda t: (len(t), t)):
        param = flat_params[key]                                   #获取当前参数

        rule_key = key[-investigate_depth:]                        #为简化规则判断，取参数路径的后几项作为规则的关键字

        if key[-1] in ['bias', 'scale']:                           #bias和scale不分片
            assert len(param.shape) == 1

        elif key[-1] == 'embedding':
            assert len(param.shape) == 2                           #检查参数维度是否是二维的，如果某个参数不符合预期，会触发AssertionError
            if param.shape[0] % mesh_model_shards != 0:            #不能均匀分片的话，将分片规则设置成P(None, 'mp')，在模型并行的维度上共享，可以确保模型在这个维度上的并行计算不会受到并行矩阵行数不均匀分片的影响
                sharding_rules[('embedding',)] = P(None, 'mp')

        else:
            if len(param.squeeze().shape) == 1:
                sharding_rules[rule_key] = None

            elif rule_key in sharding_rules:                      #确保已定义的规则的合法性
                for dim_size, rule_str in zip(
                        param.shape, sharding_rules[rule_key]):
                    assert rule_str != 'mp' or dim_size % mesh_model_shards == 0    #对模型并行，要求相应维度必须是mesh_model_shards的整数倍，以保证分片的均匀性

            elif under_attention(key) and rule_key[0][0] == 'o':  #对注意力机制的输出矩阵O设置分片规则，即模型并行维度分片，数据并行维度不分片
                sharding_rules[rule_key] = P('mp', None)

            elif under_attention(key) and rule_key[0][0] in ['q', 'k', 'v']:
                sharding_rules[rule_key] = P(None, 'mp')

            elif under_attention(key) and rule_key[0][-1] == 'o':  #关注的是规则键的最后一个元素的最后一个字符是否为o，即输出相关的参数
                sharding_rules[rule_key] = P('mp', None)

            elif under_attention(key) and rule_key[0][-1] in ['q', 'k', 'v']:
                sharding_rules[rule_key] = P(None, 'mp')

            #确定模型参数在哪个维度上进行模型并行分片
            else:
                rule_tuple = [None for _ in range(len(param.shape))]
                for dim in range(-1, -len(param.shape) - 1, -1):
                    if dim != last_dense_mp_dim and \
                            param.shape[dim] % mesh_model_shards == 0:
                        last_dense_mp_dim = dim
                        rule_tuple[dim] = 'mp'
                        break
                if all([t is None for t in rule_tuple]):
                    if last_dense_mp_dim is not None and \
                            param.shape[last_dense_mp_dim] % \
                            mesh_model_shards == 0:
                        rule_tuple[last_dense_mp_dim] = 'mp'

                sharding_rules[rule_key] = P(*rule_tuple)

    return list(sharding_rules.items())


def under_attention(flat_param_key):
    """
    检查参数键是否包含注意力机制相关的信息
    """
    for key in flat_param_key:
        if 'attention' in key.lower() or 'attn' in key.lower():
            return True
    return False
