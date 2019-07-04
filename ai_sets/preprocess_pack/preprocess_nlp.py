# -*- coding: utf-8 -*-
import os
import json
from utils.cmd_paras_check import json_parser
from utils.clean_data import clean_nlp
import pandas as pd


def preprocess_clean_nlp(allpaths, allfiles, ori_data):
    pddata = clean_nlp(ori_data, allfiles)
    # 4. 返回数据
    return pddata
