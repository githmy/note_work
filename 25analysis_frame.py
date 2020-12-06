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
from pylab import *

import scipy.stats as stats  # 该模块包含了所有的统计分析函数
import statistics as bstat


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

    # 数据的算术平均数（“平均数”）。
    bstat.mean()
    # 快速的，浮点算数平均数。
    bstat.fmean()
    # 数据的几何平均数
    bstat.geometric_mean()
    # 数据的调和均值
    bstat.harmonic_mean()
    # 数据的中位数（中间值）
    bstat.median()
    # 数据的低中位数
    bstat.median_low()
    # 数据的高中位数
    bstat.median_high()
    # 分组数据的中位数，即第50个百分点。
    bstat.median_grouped()
    # 离散的或标称的数据的单个众数（出现最多的值）。
    bstat.mode()
    # 离散的或标称的数据的众数列表（出现最多的值）。
    bstat.multimode()
    # 将数据以相等的概率分为多个间隔。
    bstat.quantiles()

    # 根据是否是全量样本调用 p开头的参数或不带p的。
    # 拟合
    # 置信
    #   单类对比相同的 期望
    #   单类对比相同的 方差
    #   二类对比相同的 期望
    #   二类对比相同的 方差
    # 数据的总体标准差
    bstat.pstdev()
    # 数据的总体方差
    bstat.pvariance()
    # 数据的样本标准差
    bstat.stdev()
    # 数据的样本方差
    bstat.variance()


def get_confidence(xdata, expect=0, std=1, prob=0.5, type=0, alpha=0.1):
    "给出x 和预期值，得出置信度 type=[-1,0,1]"
    expect = 2.6
    std = 3.1
    confid = 1 - alpha
    xdata = np.array(xdata)
    mean = xdata.mean()
    sstd = xdata.std()
    prob = stats.norm.pdf(0, expect, std)  # 在0处概率密度值
    pre = stats.norm.cdf(0, expect, std)  # 预测小于0的概率
    interval = stats.norm.interval(confid, expect, std)  # 96%置信水平的区间
    print('随机变量在0处的概率密度是{:.3f},\n    小于0的概率是{:.3f},\n    96%的置信区间是{}'.format(prob, pre, interval))
    return mean, sstd


def plot_confidence(expect=0, std=1, datanum=30):
    "给出xy 和预期值，得出置信度 type=[-1,0,1]"
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示解决方案
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示解决方案
    x = np.linspace(expect - 4 * std, expect + 4 * std, datanum)
    y = stats.norm.pdf(x, expect, std)
    plt.plot(x, y)
    plt.vlines(0, 0, 0.2, linestyles='--')
    plt.text(1.1, 0.18, '0')
    # plt.text(-2, 0.01, '下跌')
    # plt.text(2.5, 0.025, '上涨')
    plt.show()


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
    import datetime

    dtstr = '20140214213212001890'
    bb = datetime.datetime.strptime(dtstr, "%Y%m%d%H%M%S%f")
    print(bb,type(bb))
    exit()

    xdata = np.linspace(-15, 5, 30)
    mean, sstd = get_confidence(xdata, expect=0, std=1, prob=0.5, type=0, alpha=0.1)
    plot_confidence(mean, sstd)
    exit()
    main()
    print("end")
