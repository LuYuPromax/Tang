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
分布式环境下的数据的加载、处理和分发
"""

import tqdm                   
import jax                                                    #用于自动求导和GPU加速
import jax.numpy as jnp                                       #提供和numpy兼容的数组操作
from flax.training.common_utils import shard


def get_dataloader(examples, batch_size, collate_fn, do_shard):
    """
    输入：
    example:数据集
    batch_size:批大小
    collate_fn:用于将单个样本转化为批次的函数
    do_shard:是否进行数据分片
    输出：生成器，用于生成按批次处理的数据
    """
    def make_jnp(value):
        return jax.tree_util.tree_map(jnp.asarray, value)               #用于将批次中的数据转换为jax中的数组

    for i in range(0, len(examples) // batch_size):
        batch = collate_fn(
            examples=examples[i * batch_size:(i + 1) * batch_size])
        yield {
            key: shard(make_jnp(value)) if do_shard else make_jnp(value)
            for key, value in batch.items()
        }


def get_data_batches(examples,
                     batch_size,
                     collate_fn,
                     do_shard,
                     desc,
                     verbose):
    """
    获取数据的批次
    desc:进度条的描述,即在命令行中显示的文本
    verbose:True使用进度条显示,False禁用进度条
    """
    data_loader = get_dataloader(
        examples=examples,
        batch_size=batch_size,
        collate_fn=collate_fn,
        do_shard=do_shard)
    return tqdm.tqdm(                                 #tqdm.tqdm用于创建一个可视化的进度条以提供过程反馈
        data_loader,
        total=len(examples) // batch_size,
        desc=desc,
        disable=(not verbose))


def get_host_examples(examples, global_batch_size, shuffle, shuffle_rng, mesh):
    """
    根据是否洗牌以及抢当前是否处于分布式训练环境中，返回相应处理单元上的样本数据
    有助于确保每个处理单元获得不同的数据样本，以提高训练的随机性
    global_batch_size:全局批次大小=单个处理单元上批次大小×处理单元数量（并行设备数量）
    shuffle:布尔值，是否对数据进行洗牌
    shuffle_rng:洗牌时用到的数据生成器
    mesh:
    """
    if shuffle:
        shuffled_idxes = jax.random.permutation(
            key=shuffle_rng, x=len(examples))                 #使用jax.random.permutation函数生成随机排列的索引，然后根据这些索引重新排列样本
        examples = [examples[int(idx)] for idx in shuffled_idxes]

    examples = examples[:len(examples) // global_batch_size * global_batch_size]  #将数据集裁剪为可以整除全局批次大小的长度，以确保每个处理单元上的数据能够被完整地使用

    if mesh is None:
        return examples[jax.process_index()::jax.process_count()]
    else:
        return examples