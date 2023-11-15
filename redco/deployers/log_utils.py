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
定义日志和相应的输出信息
"""

import logging
import json
import jax


def get_logger(verbose, workdir):
    """
    创建或获取日志记录器，并进行相应的配置
    """
    logger = logging.getLogger('redco')      

    handler = logging.StreamHandler()             #创建一个控制台处理程序，用于将日志信息输出到控制台
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"))
    logger.addHandler(handler)             #将handler添加到日志记录器，使得日志消息能够在控制台显示

    if workdir is not None:
        handler = logging.FileHandler(filename=f'{workdir}/log.txt')
        handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S"))
        logger.addHandler(handler)         #如果文件处理程序创建成功，也添加到日志记录器中，使得日志消息能写入文件中

    logger.propagate = False        #多模块使用时，每个模块都能独立的处理日志

    #设置日志级别
    if verbose and jax.process_index() == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    #移除根记录器的所有处理程序，避免多次调用get_logger时重复添加处理程序
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    return logger


def log_info(info, title, logger, summary_writer, step):
    """
    记录日志信息，接收一些参数并将这些信息输出到日志中
    info:要记录的信息，可以是字符串或者可以转化为字符串的对象
    title:日志标题
    logger:使用上面的get_logger函数创建的日志记录器对象,负责实际记录工作
    summary_writer:用于记录摘要信息，可选参数
    step:训练步骤或迭代的索引，可选参数
    """
    info = str(info)

    if title is not None:
        if step is not None:                            #是否表明与当前步骤相关联
            title_ = f'{title} (step {step})'
        else:
            title_ = title
            step = 0

        if summary_writer is not None:          #表示使用tensorboard，则使用summary_writer.text方法将信息记录到Tensorboard中
            summary_writer.text(title, info.replace('\n', '\n\n'), step=step)
        max_len = max(max([len(t) for t in info.split('\n')]), len(title_) + 4)

        #添加一些装饰线喝标题，用于更好的阅读
        logger.info('=' * max_len)
        logger.info(f'### {title_}')
        logger.info('-' * max_len)
        for t in info.split('\n'):
            logger.info(t)
        logger.info('=' * max_len)

    else:
        logger.info(info)


def save_outputs(outputs, workdir, desc, logger, summary_writer, step):
    """
    保存模型训练产生的输出
    outputs:模型训练产生的输出，可能是一个包含各种信息的复杂结构，例如模型的预测结果等
    workdir:工作目录，即输出文件的保存路径
    desc:描述输出的字符串，用于构建输出文件名
    logger:日志记录器
    summary_writer:Tensorboard的摘要写入器,如果提供此参数,将用于记录输出的摘要信息
    step:训练步骤的索引或标识符，用于与Tensorboard中的特定步骤相关联
    """
    outputs = jax.tree_util.tree_map(str, outputs)                 #将输出中的每个元素都转换为字符串
    json.dump(outputs, open(f'{workdir}/outputs_{desc}.json', 'w'), indent=4)      #json.dump将转换后的输出保存为json文件
    logger.info(
        f'Outputs ({desc}) has been saved into {workdir}/outputs_{desc}.json.')

    if summary_writer is not None:
        samples_str = json.dumps(outputs[:10], indent=4).replace('\n', '\n\n')    #将输出的前10个元素以json格式记录为文本摘要
        summary_writer.text('outputs', samples_str, step=step)

    return f'{workdir}/outputs_{desc}.json'