# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import chardet


def read_csv_data(filename):
    # f = open(filename, "rb")
    # data_csv = f.read()
    # print(chardet.detect(data_csv)["encoding"])
    try:
        df1 = pd.read_csv(filename, header=0, encoding='utf-8')
    except:
        df1 = pd.read_csv(filename, header=0, encoding='gbk')
    return df1


def write_csv_data(datas, filename, colname):
    if os.path.isfile(filename):
        try:
            obj_pd = pd.read_csv(filename, header=0, encoding='utf-8')
        except:
            obj_pd = pd.read_csv(filename, header=0, encoding='gbk')
        obj_pd[colname] = list(datas)
    else:
        obj_pd = pd.DataFrame({colname: list(datas)})
    obj_pd.to_csv(filename, index=False, header=True, encoding="utf-8")
