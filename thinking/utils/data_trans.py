# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

import json
import pandas as pd


# 数据转化文件
def data2js(inpath, outpath):
    # 1. 读数据
    pd_data = pd.read_csv(inpath, header=0, encoding="utf8", dtype=str, sep=',')
    print(pd_data)
    print(pd_data.columns)

    # 2. 转化为js
    jsobj = []
    for indexs in pd_data.index:
        tmpobj = {}
        tmpobj["_id"] = pd_data.loc[indexs, "id"]
        tmpobj["code"] = str(pd_data.loc[indexs, "id"])
        tmpobj["level"] = int(pd_data.loc[indexs, "level"])
        tmpobj["description"] = pd_data.loc[indexs, "text"]
        tmpobj["defaultParams"] = {}
        if isinstance(pd_data.loc[indexs, "mainReviewPoints"], str):
            tmpobj["mainReviewPoints"] = pd_data.loc[indexs, "mainReviewPoints"].split(",")
        else:
            tmpobj["mainReviewPoints"] = []
        tmpobj["qType1"] = ""
        tmpobj["layout"] = []
        tmpobj["keywords"] = ""
        tmpobj["preview"] = {}
        tmpobj["steps"] = []
        jsobj.append(tmpobj)

    # 3. 写文件
    filestr = "module.exports = " + json.dumps(jsobj, ensure_ascii=False)
    print(filestr)
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(filestr)
    print("finish output js.")
