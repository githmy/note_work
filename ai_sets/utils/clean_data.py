# -*- coding: utf-8 -*-
import os
from preprocess_pack.text_clean import text_clean
from preprocess_pack.text_cut import text_cut


def clean_nlp(pdobj, allfiles):
    replace_dict = "replace_dict"
    stop_dict = "stop_dict"
    delete_dict = "delete_dict"
    user_dict = "user_dict"
    # 1. 使用字典清理所有以word_开头的字段
    replacefile = allfiles[replace_dict]
    stopfile = allfiles[stop_dict]
    deletefile = allfiles[delete_dict]
    userfile = allfiles[user_dict]

    dataset = pdobj
    # 2. 数据清理
    clean_data = text_clean(replacefile, dataset)
    data_cut = text_cut(stopfile, deletefile, userfile, clean_data)
    return data_cut


def clean_anal(pdobj, dic):
    # 1. 数值清理
    pass
    return pdobj


def clean_tsq(pdobj, dic):
    # 1. 时间序列清理
    pass
    return pdobj
