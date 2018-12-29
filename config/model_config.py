import os
import json
import six
import utils
from utils import json_to_string
from utils.cmd_paras_check import ConfigException
import copy

from ai_sets.hard import *
from ai_sets.openapi import *
from ai_sets.train import *

func_name = {
    "主题模型": None,
    "模型分类": None,
    "分词": fenci,
    "情感词识别": None,
    "情感打分": None,
    "性能参数返回": None,
    "PMI": None
}

