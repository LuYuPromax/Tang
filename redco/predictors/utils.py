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
提供一些通用的，与具体模型无关的功能，包括为样本添加索引、将样本列表组合为批次、处理预测函数的输出结果等
这些功能可以在整个框架中提高代码的可读性和复用性
"""

import copy
import numpy as np
import jax


def add_idxes(examples):
    """
    功能：为给定的样本列表examples添加一个名为__idex__的键，其值是该样本在列表中的索引
    """
    examples_upd = []
    for idx, example in enumerate(examples):       #使用enumerate遍历原始样本列表examples，获取每个样本的索引idx和样本内容example
        assert '__idx__' not in example
        # example = copy.deepcopy(example)
        example = {key: example[key] for key in example.keys()}
        example.update({'__idx__': idx})

        examples_upd.append(example)

    return examples_upd


def collate_fn_wrapper(examples, collate_fn):
    """
    将数据样本列表组合成一个批次，并在批次中添加__idx__键，其值是每个样本在列表中的索引
    """
    idxes = [example.pop('__idx__') for example in examples]
    batch = collate_fn(examples)           #将样本列表组合成一个批次
    batch['__idx__'] = np.array(idxes)

    return batch


def pred_fn_wrapper(pred_rng, params, batch, pred_fn, under_pmap):
    """
    包装预测函数，处理一些预测相关的逻辑，包括在预测结果中添加__idx__键，并根据under_pmap参数进行可能的分布式计算
    """
    idxes = batch.pop('__idx__')       #弹出批次中每个样本的索引
    preds = pred_fn(pred_rng=pred_rng, params=params, batch=batch)    #调用用户提供的pred_fn进行预测，传入随机数生成器，参数和批次
    preds = {
        'raw_preds': preds,
        '__idx__': idxes
    }

    if under_pmap:      #进行分布式计算，将预测结果从所有设备上收集到一起
        return jax.lax.all_gather(preds, axis_name='batch')
    else:
        return preds


def default_output_fn(preds):
    """
    将预测结果格式化为输出列表
    """
    batch_size = jax.tree_util.tree_leaves(preds)[0].shape[0]  #获取预测结果中第一个叶子节点的形状，并取出批次大小

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: x.shape[0] == batch_size, preds))   #确保预测结果中的每个叶子节点的第一个轴大小都等于batch_size

    outputs = []
    for i in range(batch_size):
        outputs.append(jax.tree_util.tree_map(lambda x: x[i], preds))  #将每个样本的预测结果组合成一个输出，通过使用 ax.tree_util.tree_map函数，将每个叶子节点的相应索引的值提取出来

    return outputs