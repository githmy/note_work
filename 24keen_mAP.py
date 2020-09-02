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
import xmltodict
import seaborn as sns
from keras.models import load_model

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False


def plot_bar(x, y, title='time_sequence', showiter=1):
    plt.figure()
    xin = range(len(x))
    plt.bar(left=xin, height=y, width=0.8, bottom=0.0, color='red', edgecolor='k')
    plt.xlabel("x", verticalalignment="top")
    show_inte = showiter
    s_xin = [i1 for i1 in xin if i1 % show_inte == 0]
    s_x = [i1 for id1, i1 in enumerate(x) if id1 % show_inte == 0]
    plt.xticks(s_xin, s_x, rotation=90, fontsize=10)
    plt.title(title)
    plt.show()


# 显示区间密度
def range_density(x, title="密度图"):
    plt.figure(figsize=(12, 8))
    # sns.distplot(x, bins=50, kde=False, color="red")
    # 核密度估计 + 统计柱状图
    sns.distplot(x, bins=100)
    plt.title(title)
    plt.xlabel("", fontsize=12)
    plt.show()


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


def title1():
    "统计缺陷"
    baspath = os.path.join("..", "data", "keen", "seats")
    dirpath = os.listdir(baspath)
    filenames = [os.path.join(baspath, i1) for i1 in os.listdir(baspath) if i1.endswith("xml")]
    json_static = {}
    list_static = []
    for filename in filenames:
        print(filename)
        with open(filename, 'rt', encoding="utf-8") as f:
            result = f.readlines()
            # lines = [i1.strip("\n") for i1 in result]
            xml_str = "".join(result)
            xml_parse = xmltodict.parse(xml_str)
            json_str = json.dumps(xml_parse, ensure_ascii=False)
            json_obj = json.loads(json_str, encoding="utf-8")
            # print(json_obj["annotation"]["size"])
            list_static.append(0)
            # print(json_obj["annotation"]["object"])
            for item in json_obj["annotation"]["object"]:
                if isinstance(item, str):
                    if json_obj["annotation"]["object"]["name"] not in json_static:
                        json_static[json_obj["annotation"]["object"]["name"]] = 0
                    json_static[json_obj["annotation"]["object"]["name"]] += 1
                    list_static[-1] += 1
                    break
                else:
                    if item["name"] not in json_static:
                        json_static[item["name"]] = 0
                    json_static[item["name"]] += 1
                    list_static[-1] += 1
    json_static_list = sorted(json_static.items(), key=lambda x: x[1])
    plot_bar(*zip(*json_static_list), title='缺陷类别统计', showiter=1)
    # plot_bar(range(len(list_static)), sorted(list_static), title='单品缺陷数量统计', showiter=20)
    range_density(list_static, title="缺陷单品数量密度区间图")
    # plot_bar(list(json_static.keys()), list(json_static.values()), title='缺陷类别统计')
    # plot_bar(json_static.keys(), json_static.items(), title='缺陷类别统计')


def title2():
    "错误产生的原因，提升预测准确率的方法。"
    # 由分析文件夹 images/detections_one_by_one 可以看出，许多预测的结果基本正确，只是没有达到IOU的阈值，所以降低IOU的阈值可以提高召回率。
    # 增加格点密度，提高临近的同类标签。增加不平衡类别标签的权重，来提高少样本的精度。
    pass


def title3():
    "基于给定的yolo 模型 预测，并求 mAP。保证召回率的前提下，改进模型。"
    # map 是对那个文件夹做预测？是训练集的吗？416 320
    modelpath = os.path.join("..", "keras-yolo3", "model_data", "yolo.h5")
    model = load_model(modelpath)
    model.summary()

    # 规范化图片大小和像素值
    def get_inputs(src=[]):
        pre_x = []
        for s in src:
            input = cv2.imread(s)
            input = cv2.resize(input, (150, 150))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            pre_x.append(input)  # input一张图片
        pre_x = np.array(pre_x) / 255.0
        return pre_x

    # 要预测的图片保存在这里
    baspath = os.path.join("..", "data", "keen", "seats")
    # 这个路径下有两个文件，分别是cat和dog
    test = os.listdir(baspath)
    images = [os.path.join(baspath, i1) for i1 in os.listdir(baspath) if i1.endswith("jpg")]
    # 打印后：['cat', 'dog']
    print(test)
    # 新建一个列表保存预测图片的地址
    images = []
    # 获取每张图片的地址，并保存在列表images中
    for testpath in test:
        for fn in os.listdir(os.path.join(predict_dir, testpath)):
            if fn.endswith('jpg'):
                fd = os.path.join(predict_dir, testpath, fn)
                print(fd)
                images.append(fd)
    # 调用函数，规范化图片
    pre_x = get_inputs(images)
    # 预测
    pre_y = model.predict(pre_x)



def title4():
    "分析改进后的 模型 说明异同 预测，并求 mAP。改进的空间。"
    pass


def main():
    # 1. 默认加载原数据
    pass

if __name__ == '__main__':
    title3()
    exit()
    title1()
    main()
    print("end")
