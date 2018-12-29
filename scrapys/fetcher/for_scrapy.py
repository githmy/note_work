# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from scrapy.cmdline import execute
import sys, os

# 将项目目录动态设置到环境变量中
# os.path.abspath(__file__) 获取main.py的路径
# os.path.dirname(os.path.abspath(__file__) 获取main.py所处目录的上一级目录
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# execute(['scrapy', 'crawl', 'zhihu'])
execute(['scrapy', 'crawl', 'stock'])
# execute(['scrapy', 'crawl', 'peolist'])
# execute(['scrapy', 'crawl', 'marray'])
# os.system("scrapy crawl example")
