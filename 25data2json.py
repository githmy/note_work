import os
import itertools
import numpy as np
import pytesseract
from PIL import Image
import csv
import re
import json
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib as mpl
import datetime, time
import copy
import random
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import bisect
import xmltodict
import seaborn as sns
import pandas as pd
import codecs


def main():
    # 1. 默认加载原数据
    bpath = os.path.join("g:\\", "project")
    onfile = os.path.join(bpath, "滨江森林公园公众号消息日志.xlsx")
    pddata = pd.read_excel(onfile, sheet_name='Sheet1', header=0)
    pddata1 = pddata["text"]
    onfile = os.path.join(bpath, "共青森林公园公众号消息日志.xlsx")
    pddata = pd.read_excel(onfile, sheet_name='Sheet1', header=0)
    pddata2 = pddata["text"]
    pddata_all = pd.concat([pddata1, pddata2], axis=0)
    # print(pddata_all.shape)
    # def exta(data):
    #     if isinstance(data, str):
    #         # data = json.loads(data.rstrip().rstrip("0").rstrip("〓"))
    #         print(data)
    #         data = json.loads(data.split("〓")[0])
    #         # data = data[data["msgtype"]]["content"]
    #         data = data["msgtype"]
    #     else:
    #         data = None
    #     return data
    #
    # pddata["result_json"] = pddata["result_json"].map(exta)
    # print(pddata["result_json"])
    # print(set(pddata["result_json"]))
    # 写入输出结果
    f = codecs.open(os.path.join(bpath, "key.txt"), 'w', 'utf-8')
    pddata_all = set(pddata_all)
    for item in pddata_all:
        f.write(str(item) + "\n")
    f.close()
    pass


if __name__ == '__main__':
    main()
    print("end")
