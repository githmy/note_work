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
import matplotlib.style as style
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
import numpy as np

import scipy.stats as stats  #该模块包含了所有的统计分析函数
import statistics

def static_func():
    "给出xy分布，拟合分布的参数"
    # 均匀分布
    # 二项分布
    # 负二项分布 aka 帕斯卡分布
    # 几何分布
    # 泊松分布
    # gamma分布
    # 指数分布
    # 正态分布
    # student分布
    # 卡方分布
    # F分布
    pass

def get_confidence(xydata,expect=0,std=1,prob=0.5,type=0,alpha=0.1):
    "给出xy 和预期值，得出置信度 type=[-1,0,1]"
    print(*cols)
    pass

def get_relation(*cols):
    "给出不同列，得出相关性函数指标"
    print(*cols)

def main():
    # 1. 默认加载原数据
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
