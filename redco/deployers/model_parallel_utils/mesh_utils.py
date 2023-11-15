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
    函数作用是将模型参数按照预定义的规则进行分片
    
    """
    shard_fn = pjit(
        lambda x: x, in_shardings=(None,), out_shardings=params_spec)

    with mesh:
        return shard_fn(params)


def shard_params_and_opt_state(params, params_spec, mesh, optimizer):
    def init_fn(params_):
        opt_state_ = optimizer.init(params_)
        return opt_state_, params_

    def get_opt_spec(x):
        if isinstance(x, (dict, FrozenDict,)):
            return params_spec
        return None

    params_shapes = jax.tree_util.tree_map(
        lambda x: ShapeDtypeStruct(x.shape, x.dtype), params)
    state_shapes = jax.eval_shape(init_fn, params_shapes)

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
    def param_gather_fn(param_spec):
        return pjit(
            lambda x: x, in_shardings=(param_spec, ), out_shardings=None)

    gather_fns = jax.tree_util.tree_map(
        lambda param_spec: param_gather_fn(param_spec),
        params_spec,
        is_leaf=lambda x: x is None or isinstance(x, P))

    with mesh:
        with jax.default_device(jax.devices('cpu')[0]):
            return jax.tree_util.tree_map(
                lambda gather_fn, param: jax.device_get(gather_fn(param)),
                gather_fns,
                params)


def get_sharding_rules(params, mesh_model_shards, investigate_depth=2):
    sharding_rules = {
        ('(bias|scale)',): None,
        ('embedding',): P('mp', None),
    }

    last_dense_mp_dim = None
    flat_params = flatten_dict(params)
    for key in sorted(flat_params.keys(), key=lambda t: (len(t), t)):
        param = flat_params[key]

        rule_key = key[-investigate_depth:]

        if key[-1] in ['bias', 'scale']:
            assert len(param.shape) == 1

        elif key[-1] == 'embedding':
            assert len(param.shape) == 2
            if param.shape[0] % mesh_model_shards != 0:
                sharding_rules[('embedding',)] = P(None, 'mp')

        else:
            if len(param.squeeze().shape) == 1:
                sharding_rules[rule_key] = None

            elif rule_key in sharding_rules:
                for dim_size, rule_str in zip(
                        param.shape, sharding_rules[rule_key]):
                    assert rule_str != 'mp' or dim_size % mesh_model_shards == 0

            elif under_attention(key) and rule_key[0][0] == 'o':
                sharding_rules[rule_key] = P('mp', None)

            elif under_attention(key) and rule_key[0][0] in ['q', 'k', 'v']:
                sharding_rules[rule_key] = P(None, 'mp')

            elif under_attention(key) and rule_key[0][-1] == 'o':
                sharding_rules[rule_key] = P('mp', None)

            elif under_attention(key) and rule_key[0][-1] in ['q', 'k', 'v']:
                sharding_rules[rule_key] = P(None, 'mp')

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
    for key in flat_param_key:
        if 'attention' in key.lower() or 'attn' in key.lower():
            return True
    return False
