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
import datetime, time
import copy
import random
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import bisect

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False


def plot_markcurve(x, ys, titles, lines, points, text):
    yins = [np.array(y) for y in ys]
    # xin = np.arange(0, len(ys[0]))
    xin = x
    nums = len(ys)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (nums // 7 + 1)
    # 长 宽 背景颜色
    plt.figure(figsize=(12, 6), facecolor='w')
    # plt.figure(facecolor='w')
    for n in range(nums):
        plt.plot(xin, yins[n], color=colors[n], linestyle='-', linewidth=1.2, marker="", markersize=7,
                 markerfacecolor='b', markeredgecolor='g', label=titles[n])
        for idn, line in enumerate(lines):
            plt.plot([line[0], line[2]], [line[1], line[3]], color=colors[1], linestyle='-', linewidth=1)
            plt.plot(*points[idn], "bo")
            plt.text(points[idn][0] + 0.02, points[idn][1] + 0.02, text[idn])
        plt.legend(loc='upper right', frameon=False)
    # plt.plot(xin, yin, color='r', linestyle='-', linewidth=1.2, marker="*", markersize=7, markerfacecolor='b',
    #          markeredgecolor='g')
    plt.xlabel("x", verticalalignment="top")
    plt.ylabel("y", rotation=0, horizontalalignment="right")
    # xticks = ["今天", "周五", "周六", "周日", "周一"]
    show_inte = 30
    # show_inte = 7
    s_xin = [i1 for i1 in xin if i1 % show_inte == 0]
    s_x = [i1 for id1, i1 in enumerate(x) if id1 % show_inte == 0]
    plt.xticks(s_xin, s_x, rotation=90, fontsize=10)
    # plt.xticks(xin, x, rotation=90, fontsize=5)
    # yticks = np.arange(0, 500, 10)
    # plt.yticks(yticks)
    # plt.title(title)
    # plt.grid(b=True)
    plt.savefig('{}.png'.format(titles[0]))
    # plt.show()


def getkachi_line(x, y):
    # y=kx+b
    k_thresh = 0.15
    indmin = y.argmin()
    length = len(x)
    reslist = []
    for point1id in range(length - 1):
        for point2id in range(point1id + 1, length):
            tk = (y[point2id] - y[point1id]) / (x[point2id] - x[point1id])
            b = y[point2id] - tk * x[point2id]
            # print("k b ", tk, b)
            oksig = 1
            area = 0
            for testid in range(length):
                ty = tk * x[testid] + b
                tarea = ty - y[testid]
                if tarea < 0:
                    oksig = 0
                    break
                else:
                    area += tarea
            if oksig == 1 and abs(tk) < k_thresh:
                # dis = (tk * xmin - ymin + b)/sqrt(tk^2+1)
                dis = (tk * x[indmin] - y[indmin] + b) / np.sqrt(tk * tk + 1)
                # print([point1id, point2id, area, dis])
                reslist.append([point1id, point2id, area, np.abs(dis)])
    reslist = sorted(reslist, key=lambda x: x[2])
    if len(reslist) > 0:
        reslist = reslist[0]
        # print(reslist)
        xb, yb, xc, yc, dp = x[reslist[0]], y[reslist[0]], x[reslist[1]], y[reslist[1]], reslist[3]
        return xb, yb, xc, yc, dp
    else:
        return None


def kachi_filter(relist):
    # [xa, ya, xb, yb, xc, yc, dp]
    # 卡尺过滤 检测最小纹理阈值
    minthreash = 5.0
    if len([i1 for i1 in relist if i1[-1] > minthreash]) > 0:
        nonethreash = 2.0
    else:
        nonethreash = 1.0
    # 卡尺过滤 平整部分
    relist = [i1 for i1 in relist if i1[-1] > nonethreash]
    # 删除重复
    for id in range(len(relist) - 1, 0, -1):
        siglist = [1 if one[0] == one[1] else 0 for one in zip(relist[id], relist[id - 1])]
        if len(siglist) == sum(siglist):
            del relist[id]
    # 删除 a 不在bc间的
    for id in range(len(relist) - 1, -1, -1):
        if relist[id][0] < relist[id][2] or relist[id][0] > relist[id][4]:
            relist.pop(id)
    # 删除 y深x坐标接近的
    for id in range(len(relist) - 1, 0, -1):
        if relist[id - 1][0] > relist[id][2]:
            if relist[id][-1] > relist[id - 1][-1]:
                relist.pop(id - 1)
            else:
                relist.pop(id)
    return relist


def kachi_measure(x, y):
    # 谷底数量阈值
    ratio_thresh = 0.3
    # 减去头尾
    cut = 0.1
    xlenth = len(x)
    cutlenth = int(xlenth * cut)
    x, y = x[cutlenth:xlenth - cutlenth], y[cutlenth:xlenth - cutlenth]
    # 卡尺移动
    relist = []
    kachisize = 50
    stepsize = 10
    xmin = min(x)
    xmax = max(x)
    lastx = xmin
    while 1:
        if lastx > xmax - stepsize:
            break
        # 移动后的区域
        ind_now = bisect.bisect(x, lastx) - 1
        ind_end = bisect.bisect(x, lastx + kachisize)
        xrange = x[ind_now:ind_end]
        yrange = y[ind_now:ind_end]
        # 移动后的区域的最小值
        depind = yrange.argmin()
        higind = yrange.argmax()
        ymid = (yrange.max() + yrange.min()) / 2
        ratio = len(yrange[yrange > ymid]) / len(yrange)
        # 过滤倾斜卡尺
        if ratio > ratio_thresh:
            # 根据移动后的区域的最小值 重置卡尺范围
            xa = xrange[depind]
            ya = yrange[depind]
            fixbeforeid = bisect.bisect(x, xa - kachisize / 2) - 1
            fixbeforeid = 0 if fixbeforeid < 0 else fixbeforeid
            fixafterid = bisect.bisect(x, xa + kachisize / 2)
            fixbeforeid = 0 if fixbeforeid < 0 else fixbeforeid
            # # 重置卡尺范围 再找最小值
            # print(y)
            # print(len(y))
            # print(fixbeforeid, fixafterid)
            depind = y[fixbeforeid:fixafterid].argmin()
            xa = x[fixbeforeid + depind]
            ya = y[fixbeforeid + depind]
            # 找拟合直线
            tmpout = getkachi_line(x[fixbeforeid:fixafterid], y[fixbeforeid:fixafterid])
            if tmpout is not None:
                xb, yb, xc, yc, dp = tmpout
                # tmp = [xa, ya, xb, yb, xc, yc, dp]
                tmp = [xa, ya, *tmpout]
                relist.append(tmp)
        lastx += stepsize
    return relist


def ori2horiztal(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    # reg = linear_model.LinearRegression()
    # reg.fit(x, y)
    # newy = reg.predict(x)
    # # print('Coefficients: \n', reg.coef_)
    # # # 查看均方误差
    # # print("Mean squared error: %.2f" % mean_squared_error(newy, y))
    # # # 解释方差分数:1是完美的预测
    # # print('Variance score: %.2f' % r2_score(newy, y))
    # # newx = x
    # # newy = newy.reshape(-1)
    newx = x.reshape(-1)
    newy = y.reshape(-1)
    return newx, newy


def main():
    # 1. 默认加载原数据
    baspath = os.path.join("..", "data", "轮胎检测数据存储0522")
    # , "1号胎数据"
    dirpath = os.listdir(baspath)
    paths = [os.path.join(baspath, dir) for dir in dirpath]
    filenames = []
    for txt in paths:
        filenames += [os.path.join(txt, i1) for i1 in os.listdir(txt) if i1.endswith("txt")]
    stime = time.time()
    filenames = filenames[:]
    print(len(filenames))
    for filename in filenames:
        print(filename)
        # 2. 生成数据
        with open(filename, 'rt', encoding="utf-8") as f:
            result = f.readlines()
            lines = [i1.strip("\n") for i1 in result]
        lines = lines[5:]
        coord = [i1.split(",") for i1 in lines]
        usecoord = []
        for i1 in coord:
            i1 = [i2.strip().split() for i2 in i1[:3]]
            usecoord.append([float(i1[1][0]), float(i1[2][0])])
        zpoint = sorted(usecoord, key=lambda x: x[0])
        x, y = zip(*zpoint)
        newx, newy = ori2horiztal(x, y)
        newy = -newy
        relist = kachi_measure(newx, newy)
        relist = kachi_filter(relist)
        print(relist)
        ys = []
        ys.append(newy)
        titles = [filename]
        # plot_curve(newx, ys, titles)
        # [xa, ya, xb, yb, xc, yc, dp]
        lines = [i1[2:6] for i1 in relist]
        points = [i1[0:2] for i1 in relist]
        text = [str(i1[-1]) for i1 in relist]
        plot_markcurve(newx, ys, titles, lines, points, text)
    exit()
    print("use time is {}s".format(time.time() - stime))


if __name__ == '__main__':
    np.random.seed(5)
    main()
    print("end")
