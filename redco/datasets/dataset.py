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

class Dataset:
    """
    定义一个名为'Dataset'的基类,使用'NotImplementedError'报错的方法，以表示这些方法必须由任何具体的子类来实现。
    从'Dataset'继承的具体子类将需要提供它们自己的'get_examples'和'get_size'方法的实现。
    """
    def get_examples(self, split):
        raise NotImplementedError

    def get_size(self, split):
        raise NotImplementedError
