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

import json
from glob import glob                              #用于匹配文件路径模式
import tqdm                                        #用于显示进度条的库

from .dataset import Dataset


class JsonlDataset(Dataset):
    """
    用于处理包含json行的数据集文件
    """
    def __init__(self, data_dir):
        """
        构造函数，将所有的.jsonl文件与split相关联,并将文件路径存储在_split_filenames字典中
        这里的split是表示数据集中的一个子集或划分的标识符,通常数据集会被划分为训练集train、验证集val、测试集test
        """
        self._split_filenames = {}
        for filename in glob(f'{data_dir}/*.jsonl'):
            split = filename.split('/')[-1][:-len('.jsonl')]
            self._split_filenames[split] = filename

    def __getitem__(self, split):
        """
        用于实现类的实例可以通过索引（[])访问的方法。split指定数据集部分,它使用 tqdm.tqdm 创建一个进度条，逐行读取指定拆分的 JSONL 文件，
        并将每一行的 JSON 数据解析后添加到一个列表中。最后，返回包含所有示例的列表
        """
        examples = []
        for line in tqdm.tqdm(open(self._split_filenames[split]),
                              desc=f'loading {split} examples'):
            examples.append(json.loads(line))

        return examples
