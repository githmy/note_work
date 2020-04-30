import os
import itertools
import cv2
import numpy as np
import pytesseract
from PIL import Image
import csv
import re
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False


def write2file(filename, strlist):
    with open(filename, 'wt', encoding="utf8") as f2:
        for i in strlist:
            f2.write(i + "\n")


def readfile2list(file):
    with open(file, "r", encoding="utf-8") as f:
        flist = [i1.rstrip() for i1 in f.readlines()]
    return flist


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


def statistic_num(tlist, splitlist):
    t_iterdic = {i1: 0 for i1 in tlist}
    t_splitdic = {i1: 0 for i1 in tlist}
    for id1, line in enumerate(splitlist):
        tmptoken = {i1: 0 for i1 in tlist}
        for ttken in line:
            t_iterdic[ttken] += 1
            if tmptoken[ttken] == 0:
                t_splitdic[ttken] += 1
                tmptoken[ttken] = 1
    outlist = []
    for i1 in zip(sorted(t_iterdic.items(), key=lambda x: x[0]), sorted(t_splitdic.items(), key=lambda x: x[0])):
        outlist.append([*i1[0], i1[1][1]])
    outlist = sorted(outlist, key=lambda x: -x[1])
    inx = [i1[0] for i1 in outlist]
    iny = [i1[1] for i1 in outlist]
    return inx, iny


def main():
    bapath = "C:\project\data\latexhand\human\\formulas"
    ffile = os.path.join(bapath, "formulas.norm.txt")
    tfile = os.path.join(bapath, "vocab.txt")
    fid = 0
    tid = 1600
    tid = 1
    tlist = readfile2list(tfile)
    flist = readfile2list(ffile)
    fsplitlist = [i1.split(" ") for i1 in flist]
    # f. 之前的加新的
    oldids = list(range(fid, tid))
    newids = oldids
    nowsplitlist = [fsplitlist[i1] for i1 in newids]
    # 1. 统计出 数量分布
    inx, iny = statistic_num(tlist, nowsplitlist)
    # plot_curve(inx, iny)
    # 2. 对低位的再次选取
    iterend = 70
    thresh = 5
    for loop in range(20):
        print("loop{}".format(loop))
        mapcount = {i1: 0 for i1 in inx[-iterend:]}
        choiceids = []
        for i1 in range(len(flist)):
            if i1 not in oldids:
                for token in mapcount:
                    if mapcount[token] < thresh and token in fsplitlist[i1]:
                        mapcount[token] += 1
                        choiceids.append(i1)
                        break
        # f. 再次统计新内容
        newids = oldids
        for i1 in choiceids:
            newids.append(i1)
        print(len(newids))
        nowsplitlist = [fsplitlist[i1] for i1 in newids]
        inx, iny = statistic_num(tlist, nowsplitlist)
        print(inx[-iterend:])
        print(iny[-iterend:])
        oldids = newids
    print(sorted(oldids))
    print(len(oldids))
    newnotinold2000 = [i1 for i1 in oldids if i1 not in list(range(2001))]
    oldnotinnew2000 = [i1 for i1 in range(2001) if i1 not in oldids]
    print("newnotinold2000", sorted(newnotinold2000))
    print(len(newnotinold2000))
    print("oldnotinnew2000", sorted(oldnotinnew2000))
    lenthnouse = len(oldnotinnew2000)
    print(lenthnouse)
    plot_curve(inx, iny)
    # 3.1 增加的列表
    filename = os.path.join(bapath, "add.txt")
    strlist = [str(i1) for i1 in sorted(newnotinold2000)]
    write2file(filename, strlist)
    # 3.2 token文件
    filename = os.path.join(bapath, "token.json")
    json.dump({i1[0]: i1[1] for i1 in zip(inx, iny)}, open(filename, mode='w', encoding="utf-8"), indent=4,
              ensure_ascii=False)
    # 3.3 新的训练文件
    filename = os.path.join(bapath, "train.matching.txt")
    strlist = [str(i1) + ".png " + str(i1) for i1 in sorted(oldids)]
    write2file(filename, strlist)
    print(len(strlist))
    # 3.4 新的验证文件
    filename = os.path.join(bapath, "val.matching.txt")
    strlist = [str(i1) + ".png " + str(i1) for i1 in list(sorted(oldnotinnew2000))[0:lenthnouse // 2]]
    write2file(filename, strlist)
    print(len(strlist))
    # 3.5 新的测试文件
    filename = os.path.join(bapath, "test.matching.txt")
    strlist = [str(i1) + ".png " + str(i1) for i1 in list(sorted(oldnotinnew2000))[lenthnouse // 2:]]
    write2file(filename, strlist)
    print(len(strlist))


if __name__ == '__main__':
    main()
