# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import logging


cmd_path = os.getcwd()
data_path = os.path.join(cmd_path, "data")
data_path = os.path.join(data_path, "stock")
datalogfile = os.path.join(cmd_path, 'log')
datalogfile = os.path.join(datalogfile, 'log.log')

# 创建一个logger
logger1 = logging.getLogger('logger_out')
logger1.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(datalogfile)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger1.addFilter(filter)
logger1.addHandler(fh)
logger1.addHandler(ch)


# **********************使用******************
logger1.info("error with code: %s" % coden)
