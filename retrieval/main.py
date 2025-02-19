"""Script for training the premise retriever.
"""

import os
from loguru import logger #记录日志
from pytorch_lightning.cli import LightningCLI #导入 LightningCLI 类

from retrieval.model import PremiseRetriever #从本地文件导入 PremiseRetriever 类
from retrieval.datamodule import RetrievalDataModule #从本地文件导入 RetrievalDataModule 类


class CLI(LightningCLI): #继承自LightningCLI接口
    def add_arguments_to_parser(self, parser) -> None: #重写该方法，官方源代码中定义了这个函数但是内容为空  
        parser.link_arguments("model.model_name", "data.model_name") #把"model.model_name"和"data.model_name"链接在一起使得在命令行中设置一个参数的值会自动更新另一个参数的值
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")#会将当前进程的进程 ID 作为信息输出到日志中，使用 f-string 格式化字符串将进程 ID 嵌入到输出信息中，格式为 "PID: [进程 ID]"。
    cli = CLI(PremiseRetriever, RetrievalDataModule)#主要内容是model的PremiseRetriever类，datamodule的RetrievalDataModule类
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
