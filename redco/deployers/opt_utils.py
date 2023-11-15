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
用于获取学习率调度函数
"""

import optax

def get_lr_schedule_fn(schedule_type,
                       total_train_steps,
                       warmup_steps,
                       init_learning_rate,
                       learning_rate,
                       end_learning_rate):
    """
    schedule_type:学习率调度的类型，支持linear和cosine两种类型
    total_train_steps:总的训练步数
    warmup_steps：学习率预热的步数
    init_learning_rate:初始学习率
    learning_rate:学习率的最大值（经过预热后达到的学习率）
    end_learning_rate:学习率的最终值
    """
    warmup_fn = optax.linear_schedule(             #线性学习率预热函数
        init_value=init_learning_rate,
        end_value=learning_rate,
        transition_steps=warmup_steps)

    if schedule_type == 'linear':
        decay_fn = optax.linear_schedule(          #线性衰减函数，在total_train_steps-warmup_steps步后线性过渡到end_learning_rate
            init_value=learning_rate,
            end_value=end_learning_rate,
            transition_steps=total_train_steps - warmup_steps)
    elif schedule_type == 'cosine':                #余弦衰减函数
        decay_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_train_steps - warmup_steps,
            alpha=end_learning_rate / learning_rate)
    else:
        raise ValueError(f'lr schedule_type={schedule_type} not supported now.')

    lr_schedule_fn = optax.join_schedules(                                #将warmup_fn和decay_fn组合成一个整体的学习率调度函数lr_schedule_fn
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])

    return lr_schedule_fn
