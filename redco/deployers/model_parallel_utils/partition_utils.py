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
#
#  adapted from https://github.com/google-research/google-research/blob/fce923e9dad97cd67492c2a65b9ecdc4b2495204/flax_models/t5x/partitions.py
"""Utilities for constructing PyTrees of PartitionSpecs."""
"""
定义了用于对模型参数进行分区的函数
这些分区规则可以根据用户需求定义，以满足不同的模型结构和分布式训练的需求
"""
import re                 #python标准库中的正则表达式模块，在这里用于支持用户定义的正则表达式规则

from flax.core.frozen_dict import freeze             #用于表示不可变的字典，在这里被用来包装最终的分区规则结果
from flax.traverse_util import flatten_dict, unflatten_dict


# Sentinels
_unmatched = object()         #用作一个标记，表示未匹配到任何规则


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    """
    用于检查一组正则表达式是否匹配给定的一组字符串
    qs:一组正则表达式的字符串列表，表示匹配规则
    ks:一组字符串的元组，表示待匹配的字符串
    """
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))          #将qs编译为正则表达式对象，添加$以强制完全匹配
    for i in range(len(ks) - len(qs) + 1):                       #对于每个窗口，使用编译后的正则表达式对象匹配对应的子字符串
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):                             #任何窗口的所有正则表达式都匹配成功
            return True
    return False


def _replacement_rules(rules):
    """
    辅助函数，用于生成一个替换函数，该函数根据给定的规则替换输入的键值对
    rules:一个规则列表，每个规则是一个包含正则表达式和替换值的元组
    """
    def replace(key, val):
        for rule, replacement in rules:           #遍历规则列表，如果键匹配规则中任何一个正则表达式，则返回相应替换之
            if _match(rule, key):
                return replacement
        return val

    return replace              #注意返回值是一个函数


def set_partitions(in_dict, rules):
    """
    起主要作用的函数，用于根据给定的分区规则对参数字典进行分区
    in_dict:输入的参数字典，其中包含了需要进行分区的参数
    rules:规则列表,每个规则都是一个包含正则表达式和替换值的元组
    """
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}     #键值未in_dict扁平化版本的键，初始值为_ubmatched
    result = {k: replace(k, v) for k, v in initd.items()}      #使用替换函数替换键值，得到新的字典
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))            #返回冻结的，未分区的字典
