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

import os
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard_prng_key
from flax.core.frozen_dict import unfreeze
from flax.serialization import (
    msgpack_serialize, msgpack_restore, from_state_dict, to_state_dict)

from .data_utils import get_host_examples, get_data_batches
from .opt_utils import get_lr_schedule_fn
from .log_utils import get_logger, log_info, save_outputs
from .model_parallel_utils.mesh_utils import (
    get_mesh,
    shard_params_and_opt_state,
    shard_params,
    gather_params_to_cpu,
    get_param_spec,
    get_sharding_rules)


class Deployer:
    def __init__(self,
                 jax_seed,
                 n_model_shards=1,
                 verbose=True,
                 workdir=None,
                 run_tensorboard=False,
                 run_wandb=False):
        """
        Deployer类的构造函数,用于创建Deployer类的实例
        jax_seed:JAX的随机数种子
        n_model_shards:模型并行的分片数,默认为1
        verbose:是否启用详细输出,默认为True
        workdir:保存工作文件的目录路径,默认为None
        run_tensorboard:是否运行TensorBoard,默认为Flase
        run_wandb:是否运行WandB,默认为False
        """
        if workdir is not None:                      #创建目录，exist_ok=True表示如果目录已存在则不会引发错误
            os.makedirs(workdir, exist_ok=True)

        self._verbose = verbose                      #将传入参数存储为实例属性
        self._workdir = workdir
        self._logger = get_logger(verbose=verbose, workdir=workdir)        #创建日志记录器

        if run_wandb:                                #wandb是一个用于机器学习实验管理和结果可视化的工作，提供记录与跟踪、可视化、协作共享、超参数优化等功能
            import wandb
            self._wandb_log_fn = wandb.log
        else:
            self._wandb_log_fn = None

        if run_tensorboard and jax.process_index() == 0:     #需要当前进程索引为0才导入，可能是为了避免每个进程都尝试导入，从而避免不必要的重复导入和资源消耗
            from flax.metrics import tensorboard
            self._summary_writer = tensorboard.SummaryWriter(workdir)
        else:
            self._summary_writer = None

        self._rng = jax.random.PRNGKey(seed=jax_seed)        #使用传入的jax_seed创建随机数生成器的种子
        self._mesh = get_mesh(n_model_shards=n_model_shards) #调用get_mesh方法获得模型的拓扑结构

    def process_batch_size(self, per_device_batch_size):
        """
        根据每个设备的批次大小和总的训练设备数量来计算总的批次大小
        per_device_batch_size:每个设备上的批次大小
        """
        if self._mesh is None:
            batch_size = per_device_batch_size * jax.local_device_count()
            global_batch_size = batch_size * jax.process_count()          #还需要考虑进程数量
        else:
            batch_size = per_device_batch_size * self._mesh.shape['dp']   #模型并行维度上的设备的数量
            global_batch_size = batch_size

        return batch_size, global_batch_size

    def get_model_input_batches(self,
                                examples,
                                per_device_batch_size,
                                collate_fn,
                                shuffle,
                                shuffle_rng,
                                desc):
        """
        用于获取模型的输入批次
        examples:数据集,再dataset中提过,train、val、test三类
        per_device_batch_size:每个设备上的批次大小
        collate_fn:用于整理批次的函数,负责将不同设备上收集到的数据整理成模型能够接受的形式
        shuffle:布尔值，决定是否洗牌
        shuffle_rng:洗牌用的随机数生成器
        desc:描述性字符串,用于标识当前操作,用于日志记录
        """
        batch_size, global_batch_size = self.process_batch_size(            #计算每个设备的批次大小和全局批次大小
            per_device_batch_size=per_device_batch_size)

        examples = get_host_examples(         #从examples中获取主机上的示例
            examples=examples,
            global_batch_size=global_batch_size,
            shuffle=shuffle,
            shuffle_rng=shuffle_rng,
            mesh=self._mesh)

        return get_data_batches(            #根据数据示例（总的数据量），划分出具有指定batch_size的批次
            examples=examples,
            batch_size=batch_size,
            collate_fn=collate_fn,
            do_shard=(self.mesh is None),
            desc=f'{desc} (global_batch_size = {global_batch_size})',
            verbose=self._verbose)

    def process_batch_preds(self, batch_preds_with_idxes): 
        """
        在推理阶段处理批次数据的预测结果，以确保最终得到的预测结果是正确的形状
        batch_preds_with_idxes:包含了批次信息的原始预测结果
        """
        if self._mesh is None:
            batch_preds_with_idxes = jax.tree_util.tree_map(              #对于每个张量x，获取原始预测结果x[0]，去除分布式训练中可能加入的分片信息
                lambda x: x[0], batch_preds_with_idxes)

            batch_preds = batch_preds_with_idxes['raw_preds']
            idxes = batch_preds_with_idxes['__idx__']

            preds = jax.tree_util.tree_map(
                lambda t: t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]),
                batch_preds)
            idxes = idxes.reshape(-1)
            idxes_argsort = jnp.argsort(idxes, axis=None)

            return jax.tree_util.tree_map(lambda t: t[idxes_argsort], preds)
        else:                                                                   #直接返回批次的原始预测结果，因为在分布式环境中，预测结果可能会被跨多个设备分片，后面定义专用的处理脚本
            return batch_preds_with_idxes['raw_preds']

    def process_to_run_model(self, x, is_prng_key=False):
        """
        函数功能是准备输入以在模型上运行，返回经过处理的输入，确保在分布式设置中正确使用
        x:输入数据或随机数种子
        is_prng_key:布尔值,指示x是否是伪随机数生成器密钥,默认为Flase
        """
        if self._mesh is None:
            if is_prng_key:  #在数据并行下，使用replicate对输入数据复制，或者使用shard_prng_key函数对伪随机数生成器密钥进行分片
                return shard_prng_key(x)
            else:
                return replicate(x)      
        else:
            return x

    def process_to_deliver(self, x):
        """
        准备交付模型输出，返回经过处理的输出
        x:模型的输出
        """
        if self._mesh is None:
            return unreplicate(x)   #unreplicate用于从replicate函数复刻到所有设备的参数中，提取出一个主设备上的副本
        else:
            return x

    def get_lr_schedule_fn(self,
                           train_size,
                           per_device_batch_size,
                           n_epochs,
                           learning_rate,
                           schedule_type='linear',
                           warmup_rate=0.,
                           warmup_steps=None,
                           init_learning_rate=0.,
                           end_learning_rate=0.):
        """
        用于获取学习率调度函数，该函数定义了在训练期间如何调整学习率
        train_size:整个训练集大小
        per_device_batch_size:每个设备的批量大小
        n_epochs:训练的总轮数
        learning_rate:初始学习率
        schedule_type:学习率调度的类型,可以是linear和cosine
        warmup_rate:学习率预热的步数,如果不提供,则根据总的训练步数和预热比例计算
        init_learning_rate:预热阶段初始学习率
        end_learning_rate:预热阶段结束时的学习率
        """
        _, global_batch_size = self.process_batch_size(
            per_device_batch_size=per_device_batch_size)
        total_train_steps = n_epochs * (train_size // global_batch_size)

        if warmup_steps is None:
            warmup_steps = int(total_train_steps * warmup_rate)

        return get_lr_schedule_fn(          #注意这里是调用了opt_utils中的同名函数
            schedule_type=schedule_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            init_learning_rate=init_learning_rate,
            learning_rate=learning_rate,
            end_learning_rate=end_learning_rate)

    def get_sharding_rules(self, params):
        """
        获取参数的分片规,并在日志中进行记录,以便后续的分布式训练中进行参考和调试
        params:包含模型参数的字典,其中键是参数的名称,值是对应的jax.numpy数组
        """
        if self._mesh is None:
            return None
        else:
            sharding_rules = get_sharding_rules(                           #函数调用
                params=params, mesh_model_shards=self._mesh.shape['mp'])

            self.log_info(
                info='\n'.join([f'{t}' for t in sharding_rules]),
                title='Sharding rules')

            return sharding_rules

    def get_params_spec(self, params, params_sharding_rules):
        """
        调用get_param_spec函数,并将模型参数和参数分片规则传递给该函数,返回结果
        get_param_spec函数是在之前的分析中讨论的一个函数,它的作用是根据模型参数和参数分片规则生成参数的规范说明(parameter spec)。
        规范说明是一个描述模型参数形状和数据类型的结构，它保留了模型参数的层次结构
        """
        return get_param_spec(
            params=params, params_sharding_rules=params_sharding_rules)

    def shard_params(self, params, params_spec):
        """
        根据模型参数和模型参数说明对模型参数进行分片
        """
        return shard_params(
            params=params, params_spec=params_spec, mesh=self._mesh)

    def shard_params_and_opt_state(self, params, params_spec, optimizer):
        """
        根据模型参数的规范说明和mesh对模型参数和优化器状态进行分片。分片是在数据并行和模型并行维度上进行的
        """
        return shard_params_and_opt_state(
            params=params,
            params_spec=params_spec,
            mesh=self._mesh,
            optimizer=optimizer)

    def run_model_step(self, step_fn, input_args):
        """
        在模型并行的环境中执行模型训练的一个步骤
        step_fn:包含模型训练步骤的函数
        input_args:传递给step_fn函数的参数
        """
        if self._mesh is None:
            return step_fn(*input_args)
        else:
            with self._mesh:
                return step_fn(*input_args)

    def gen_rng(self):
        """
        生成新的随机数生成器并更新内部的RNG
        使用jax.random.split函数将当前的RNG(self._rng)分割成两个新的RNG
        """
        self._rng, new_rng = jax.random.split(self._rng)
        return new_rng

    def log_info(self, info, title=None, step=None):
        """"
        用于在训练过程中记录日志信息,如损失值、评估指标等
        """
        if jax.process_index() == 0:                      #在进程索引为0的节点上记录日志信息
            log_info(
                info=info,
                title=title,
                logger=self._logger,
                summary_writer=self._summary_writer,
                step=step)

    def log_metrics(self, metrics, step):
        """
        记录指标信息，将其写入摘要中，同时可选地将其写入WandB
        metrics:包含要记录的指标的字典
        step:步骤数，表示记录指标的时间步
        """
        if self._summary_writer is not None:
            for metric_name, value in metrics.items():
                self._summary_writer.scalar(metric_name, value, step=step)

        if self._wandb_log_fn is not None and jax.process_index() == 0:
            self._wandb_log_fn(metrics, step)

    def save_outputs(self, outputs, desc, step):
        """
        保存模型输出到指定工作目录中
        outputs:要保存的模型输出
        desc:输出描述，用于命名保存的文件
        step:保存模型输出的时间步
        """
        if self._workdir is not None and jax.process_index() == 0:
            save_outputs(                                   #log_utils.py中的save_outputs函数
                workdir=self._workdir,
                outputs=outputs,
                desc=desc,
                step=step,
                logger=self._logger,
                summary_writer=self._summary_writer)

    def load_params(self, filepath):
        """
        从磁盘加载模型参数
        """
        with jax.default_device(jax.devices('cpu')[0]):              #默认设备CPU
            params = msgpack_restore(open(filepath, 'rb').read())    #从文件中还原被序列化的参数
            params = jax.tree_util.tree_map(jnp.asarray, params)     #将参数树的所有叶子节点转换为jnp.asarray，确保参数数据类型为jax数组

        self.log_info(f'params loaded from {filepath}')
        return params

    def load_opt_state(self, ckpt_dir, desc, target):
        """
        从磁盘加载优化器状态
        ckpt_dir:检查点保存的目录
        desc:描述检查点的字符串
        target:优化器状态目标
        """
        if self._mesh is None:
            filepath = f'{ckpt_dir}/opt_state_{desc}.msgpack'
            opt_state = msgpack_restore(open(filepath, 'rb').read())
            opt_state = from_state_dict(target=target, state=opt_state)
            opt_state = replicate(opt_state)          #复制到所有设备上
        else:
            n_processes_per_model = max(
                1, self._mesh.shape['mp'] // jax.local_device_count())        #模型并行的进程数
            ckpt_process_idx = jax.process_index() % n_processes_per_model    #计算进程索引
            filepath = (f'{ckpt_dir}/opt_state_{desc}'
                        f'_process_{ckpt_process_idx}.msgpack')
            opt_state = msgpack_restore(open(filepath, 'rb').read())
            opt_state = from_state_dict(target=target, state=opt_state)

        return opt_state

    def save_params(self,
                    params,
                    ckpt_dir,
                    desc,
                    params_sharding_rules=None):
        """
        保存模型参数
        """
        if self._mesh is None:
            params = unreplicate(params)           #将模型参数从所有设备的复制中还原
        else:                                      #获取分片规则，并将参数收集到cpu上
            params_spec = self.get_params_spec(
                params=params, params_sharding_rules=params_sharding_rules)
            params = gather_params_to_cpu(
                params=params, params_spec=params_spec, mesh=self._mesh)

        if jax.process_index() == 0:               #将参数序列化为msgpack格式，保存并记录日志
            filepath = f'{ckpt_dir}/params_{desc}.msgpack'
            open(filepath, "wb").write(msgpack_serialize(unfreeze(params)))
            self.log_info(f'params saved into {filepath}')

    def save_opt_state(self, opt_state, ckpt_dir, desc):
        if self._mesh is None:
            if jax.process_index() == 0:
                opt_state = to_state_dict(unreplicate(opt_state))       #从所有设备的复制中还原，并转化为字典格式

                filepath = f'{ckpt_dir}/opt_state_{desc}.msgpack'
                open(filepath, "wb").write(
                    msgpack_serialize(unfreeze(opt_state)))
                self.log_info(f'opt_state saved into {filepath}')
        else:
            assert (jax.local_device_count() % self._mesh.shape['mp'] == 0 or
                    self._mesh.shape['mp'] % jax.local_device_count() == 0)
            n_processes_per_model = max(
                1, self._mesh.shape['mp'] // jax.local_device_count())

            if jax.process_index() < n_processes_per_model:   #检查进程索引为0或者小于每个模型的进程数，保证只有一个进程负责保存优化器状态
                opt_state = to_state_dict(opt_state)

                filepath = (f'{ckpt_dir}/opt_state_{desc}'
                            f'_process_{jax.process_index()}.msgpack')
                open(filepath, "wb").write(
                    msgpack_serialize(unfreeze(opt_state)))
                self.log_info(f'opt_state saved into {filepath}')

    def save_rng(self, ckpt_dir, desc):
        """
        保存随机数生成器状态，保证训练过程的随机性，也方便重现
        """
        if jax.process_index() == 0:
            jnp.save(f'{ckpt_dir}/rng_{desc}.npy', self._rng)


    """
    通过过@property装饰器定义了一个只读的属性。这意味着在类的外部，可以通过访问deployer_instance.mesh来获取_mesh属性的值，
    但不能直接对 deployer_instance.mesh 进行赋值
    """
    @property
    def mesh(self):
        return self._mesh

    @property
    def workdir(self):
        return self._workdir
