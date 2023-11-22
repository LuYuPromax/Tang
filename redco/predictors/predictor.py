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

from functools import partial
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from flax.core.frozen_dict import freeze
from .utils import \
    add_idxes, collate_fn_wrapper, pred_fn_wrapper, default_output_fn


class Predictor:
    def __init__(self,
                 deployer,
                 collate_fn,
                 pred_fn,
                 output_fn=None,
                 params_sharding_rules=None):
        """
        初始化实例变量，处理输出函数，完成部署参数设置和参数初始化，包装预测函数，注意partial函数
        """
        self._deployer = deployer        #部署器对象，负责模型的部署和运行
        self._collate_fn = partial(collate_fn_wrapper, collate_fn=collate_fn)     #数据组合函数的包装，用于将数据组合成批次

        self._params_sharding_rules = params_sharding_rules   #参数分片规则
        self._pred_fn = partial(                   #包装后的预测函数，用于执行实际的模型预测
            pred_fn_wrapper,
            pred_fn=pred_fn,
            under_pmap=self._deployer.mesh is None)
        self._params_spec = None          #模型参数规范
        self._p_pred_step = None          #预测时的时间步

        if output_fn is None:             #输出函数
            self._output_fn = default_output_fn
        else:
            self._output_fn = output_fn

    def setup_running_step(self,
                           pred_fn,
                           dummy_batch,
                           params,
                           params_sharding_rules):
        """
        设置运行步骤，根据部署器的情况选择使用jax.pmap或pjit进行批处理
        """
        if self._deployer.mesh is None:          #不在分布式环境下，可以使用jax.pmap进行批处理
            self._p_pred_step = jax.pmap(pred_fn, axis_name='batch')
        else:
            data_spec = {                        #指定数据划分方式
                key: P(*(('dp',) + (None,) * (len(value.shape) - 1)))
                for key, value in dummy_batch.items()
            }

            self._params_spec = self._deployer.get_params_spec(                 #获取模型参数划分方式规范
                params=params, params_sharding_rules=params_sharding_rules)

            self._p_pred_step = pjit(
                pred_fn,
                in_shardings=(None, self._params_spec, data_spec),   #第一个参数无需划分，模型参数按照self._params_spec进行划分，数据按照data_spec划分
                out_shardings=None)

    def predict(self,
                examples,
                per_device_batch_size,
                params,
                params_meshed=False,
                desc=''):
        """
        用于执行预测任务，执行预测前进行一系列数据处理、模型参数处理、批处理操作
        examples:数据列表
        per_device_batch_size:每设备批次大小
        params:模型参数
        """
        params = freeze(params)     #冻结参数，确保参数是不可变的，以便在后续的处理中不会被修改

        raw_n_inputs = len(examples)      #原始输入数据样本数量
        _, global_batch_size = self._deployer.process_batch_size(     #_表示一个占位符，该变量的值后续并不会被使用到，因此我们不关心其取值
            per_device_batch_size=per_device_batch_size)       #根据每设备批次大小计算全局批次大小
        examples = examples + [examples[0]] * (global_batch_size - 1)   #将examples复制并补齐，确保批次大小是全局批次大小的整数倍 
        examples = add_idxes(examples=examples)  

        params = self._deployer.process_to_run_model(params)  #参数处理

        data_batches = self._deployer.get_model_input_batches(         #获取数据批次
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            shuffle_rng=None,
            desc=f'Predicting ({desc})')

        preds = []
        for batch in data_batches:
            if self._p_pred_step is None:           #如果运行步骤尚未设置，则设置运行步骤
                self.setup_running_step(
                    pred_fn=self._pred_fn,
                    dummy_batch=batch,
                    params=params,
                    params_sharding_rules=self._params_sharding_rules)

            if (self._params_spec is not None) and (not params_meshed):   #如果存在参数分片规则并且尚未切分，调用shard_params函数进行参数分片
                params = self._deployer.shard_params(
                    params=params, params_spec=self._params_spec)
                params_meshed = True

            pred_rng = self._deployer.process_to_run_model(         #生成用于预测的随机数
                self._deployer.gen_rng(), is_prng_key=True)

            batch_preds_with_idxes = self._deployer.run_model_step(   #执行定义运行步骤，得到带有序列的预测结果
                step_fn=self._p_pred_step,
                input_args=(pred_rng, params, batch))

            batch_preds = self._deployer.process_batch_preds(       #处理预测结果，得到带索引的预测结果？？
                batch_preds_with_idxes=batch_preds_with_idxes)
            batch_preds = jax.tree_util.tree_map(np.asarray, batch_preds)   #预测结果转化为numpy数组

            batch_preds = self._output_fn(batch_preds)
            assert isinstance(batch_preds, list) and \
                   len(batch_preds) == global_batch_size    #断言检查，确保输出函数返回的结果是一个列表，且长度等于全局批次大小
            preds.extend(batch_preds)

        return preds[:raw_n_inputs]