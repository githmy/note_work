import os
import itertools
import cv2
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
import datetime

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False


def getdata():
    datajson = {
        "客户初始数": 20000,
        "月嫂初始数": 1000,
        "客户日增数": 500,
        "月嫂日增数": 100,
    }
    return datajson


def plot_curve(x, y):
    y = y
    yin = np.array(y)
    xin = np.arange(0, len(y))
    # 长 宽 背景颜色
    plt.figure(figsize=(24, 12), facecolor='w')
    # plt.figure(facecolor='w')
    plt.plot(xin, yin, color='r', linestyle='-', linewidth=1.2, marker="*", markersize=7, markerfacecolor='b',
             markeredgecolor='g')
    plt.xlabel("tokenid", verticalalignment="top")
    plt.ylabel("数量", rotation=0, horizontalalignment="right")
    # xticks = ["今天", "周五", "周六", "周日", "周一"]
    plt.xticks(xin, x)
    yticks = np.arange(0, 500, 10)
    plt.yticks(yticks)

    plt.grid(b=True)
    plt.show()


def derive(data):
    # 1. 根据订单详情，排班月嫂
    data = {}
    cash = {}
    schedule = {}
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y%m%d'))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=1))

    xs = ["2017%02d01" % t for t in range(1, 13) if t % 2 == 0]
    xlabels = [t + '日期' for t in xs]
    xs = [datetime.datetime.strptime(t, '%Y%m%d') for t in xs]
    ys = [t * 2 for t in range(1, 13) if t % 2 == 0]

    ax.axes.set_xticks(xs)
    ax.axes.set_xticklabels(xlabels, rotation=40, fontproperties=myfont)
    return cash, schedule


def main():
    datajson = getdata()
    filename = os.path.join(bapath, "test.matching.txt")
    strlist = [str(i1) + ".png " + str(i1) for i1 in list(sorted(oldnotinnew2000))[lenthnouse // 2:]]
    write2file(filename, strlist)
    print(len(strlist))


if __name__ == '__main__':
    main()
