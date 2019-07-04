# -*- coding: utf-8 -*-
import datetime
import os
import json
from utils.cmd_paras_check import json_parser
import pandas as pd
from utils.path_tool import makesurepath
import re


def get_basic(args, rootpath):
    """
    路径 文件 json 获取
    :param args: 命令行参数
    :param rootpath: 根目录
    :return: 
    """
    project_nkey = "project"
    json_fkey = "json_file"
    # 1. 得到路径集合
    allpaths = get_contents_path(args, rootpath)
    # 2. 得到文件名集合
    allfiles = get_file_path(allpaths, args)
    # 3. json补足
    jsobj = {project_nkey: args.project, json_fkey: args.jsonfile}
    with open(allfiles[json_fkey], encoding="utf-8") as f:
        ori_json = json.load(f)
    standjson = json_add(ori_json, jsobj)
    # 4. json格式判断
    standjson = json_parser(standjson)
    # 5. 字典文件
    allfiles = load_jieba_dict(allpaths, allfiles, standjson)
    return allpaths, allfiles, standjson


def load_jieba_dict(allpaths, allfiles, standjson):
    """
    根据json 判断加字典
    :param allpaths : 所有路径
    :param allfiles: 所有文件
    :param standjson: 标准json
    :return: 所有文件
    """
    input_dkey = "inputpath"

    json_fkey = "json_file"

    replacefile = "replace_dict"
    stopfile = "stop_dict"
    deletefile = "delete_dict"
    userfile = "user_dict"
    if standjson["purpose"]["name"] == "nlp_label":
        if "dict" in standjson["purpose"]:
            if standjson["purpose"]["dict"] == "self":
                allfiles[replacefile] = os.path.join(allpaths[input_dkey], standjson[json_fkey] + "_replace_dict_.txt")
                allfiles[stopfile] = os.path.join(allpaths[input_dkey], standjson[json_fkey] + "_stop_dict_.txt")
                allfiles[deletefile] = os.path.join(allpaths[input_dkey], standjson[json_fkey] + "_delete_dict_.txt")
                allfiles[userfile] = os.path.join(allpaths[input_dkey], standjson[json_fkey] + "_user_dict_.txt")
            elif standjson["purpose"]["dict"] == "default":
                allfiles[replacefile] = os.path.join("dict", "_replace_dict_.txt")
                allfiles[stopfile] = os.path.join("dict", "_stop_dict_.txt")
                allfiles[deletefile] = os.path.join("dict", "_delete_dict_.txt")
                allfiles[userfile] = os.path.join("dict", "_user_dict_.txt")
            else:
                allfiles[replacefile] = os.path.join("dict", "_replace_dict_.txt")
                allfiles[stopfile] = os.path.join("dict", "_stop_dict_.txt")
                allfiles[deletefile] = os.path.join("dict", "_delete_dict_.txt")
                allfiles[userfile] = os.path.join("dict", "_user_dict_.txt")
        else:
            allfiles[replacefile] = os.path.join("dict", "_replace_dict_.txt")
            allfiles[stopfile] = os.path.join("dict", "_stop_dict_.txt")
            allfiles[deletefile] = os.path.join("dict", "_delete_dict_.txt")
            allfiles[userfile] = os.path.join("dict", "_user_dict_.txt")

    return allfiles


def json_add(ori_json, jsobj):
    """
    键名判断补足
    :param filename: 
    :param jsobj: 
    :return: 添加后的json
    """
    # [ori_json.__setitem__(item, jsobj[item]) for item in jsobj if item not in ori_json]
    for item in jsobj:
        ori_json[item] = jsobj[item]
    return ori_json


def get_contents_path(args, rootpath):
    """
    生成所有文件路径集
    :param args:命令行参数 
    :param rootpath:根路径名 
    :return: 返回json路径集
    """
    modelname = args.model
    projectname = args.project
    jsonfile = args.jsonfile
    if modelname is None:
        modelname = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    inputpath = os.path.join(rootpath, 'input', projectname)
    jsonpath = os.path.join(rootpath, 'conf', projectname)
    logpath = os.path.join(rootpath, 'logs', projectname)
    modelpath = os.path.join(rootpath, 'models', projectname, jsonfile, modelname)
    outputpath = os.path.join(rootpath, 'output', projectname)
    makesurepath(inputpath)
    makesurepath(jsonpath)
    makesurepath(logpath)
    makesurepath(modelpath)
    makesurepath(outputpath)
    return {
        "inputpath": inputpath,
        "jsonpath": jsonpath,
        "logpath": logpath,
        "modelpath": modelpath,
        "outputpath": outputpath,
    }


def get_file_path(allpaths, args):
    """
    获取文件原始数据, json 数据    
    :param allpaths:所有路径
    :param args: 命令行参数
    :return: filejson
    """
    # 1. 判断参数格式
    # 2. 路径键名
    json_dkey = "jsonpath"
    input_dkey = "inputpath"
    model_dkey = "modelpath"
    log_dkey = "logpath"
    out_dkey = "outputpath"

    # 2. 文件具体路径
    # 2.1 文件键名
    json_fkey = "json_file"
    train_fkey = "train_file"
    predict_fkey = "predict_file"
    predict_res_fkey = "predict_res_file"
    tfidf_model_fkey = "tfidf_model_file"
    sk_model_fkey = "sk_model_file"
    optimal_model_fkey = "optimal_model_file"
    model_fkey = "model_file"

    # 2.2 文件键 内容
    filejson = {}
    filejson[json_fkey] = os.path.join(allpaths[json_dkey], args.jsonfile + '.json')
    filejson[train_fkey] = os.path.join(allpaths[input_dkey], args.jsonfile + '_train.csv')
    filejson[predict_fkey] = os.path.join(allpaths[input_dkey], args.jsonfile + '_predict.csv')
    filejson[predict_res_fkey] = os.path.join(allpaths[input_dkey], args.jsonfile + '_predict_result.csv')

    return filejson


def get_words_labels(data_cols, standjson):
    words_cols_list = []
    word_list = [i for i in data_cols if i.startswith("word_") and i.endswith("__cut_")]
    re_cut = re.compile(r'__cut_$')
    for word in word_list:
        wordres = re.sub(re_cut, '', word)
        for labelname in standjson["label_map"][wordres]:
            words_cols_list.append([wordres, labelname])
    return words_cols_list
