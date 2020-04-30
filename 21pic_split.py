import os

import cv2
import numpy as np
import pytesseract
from PIL import Image
import csv
import re
import json
from utils.path_tool import makesurepath
import skimage
import copy


def getregion(region):
    y1 = min(region["all_points_y"])
    y2 = max(region["all_points_y"])
    x1 = min(region["all_points_x"])
    x2 = max(region["all_points_x"])
    return x1, x2, y1, y2


def modlist(inlist, xl, xr):
    outlist = []
    rb = xr - xl
    for i1 in inlist:
        tnum = i1 - xl
        if tnum < 0:
            tnum = 0
        if tnum > rb:
            tnum = rb
        outlist.append(tnum)
    return outlist


def modxylist(inxlist, inylist, xl, xr, yl, yr):
    lenth0 = len(inxlist)
    print(len(inxlist))
    inxlist.append(inxlist[0])
    inylist.append(inylist[0])
    outxlist = []
    outylist = []
    xrb = xr - xl
    yrb = yr - yl
    for id, (px, py) in enumerate(zip(inxlist, inylist)):
        # px, py = inxlist[id], inylist[id]
        if id < lenth0:
            print(id, px, py)
            tx = px - xl
            ty = py - yl
            if tx < 0:
                tx = 0
            if tx > xrb:
                tx = xrb
            outxlist.append(tx)
            outylist.append(ty)
    print(len(outxlist))
    exit()
    return outxlist, outylist


def main():
    # 0. 根据 试卷级图片 和 rec json, 拆图片 和新json
    tablestr = "title"
    latexstr = "markr"
    normalstr = "marke"
    basepath = os.path.join("..", "data", "autoex")
    oripath = os.path.join(basepath, "oripic")
    makesurepath(oripath)
    newpath = os.path.join(basepath, "newpic")
    makesurepath(newpath)
    # 1. 读取 试卷目录 和 rec json
    orlist = os.listdir(oripath)
    orjson = json.load(open(os.path.join(basepath, "train-bak.json"), encoding="utf8"))
    newfile = os.path.join(basepath, "split.json")
    newjson = {}
    newjson["_via_attributes"] = orjson["_via_attributes"]
    newjson["_via_settings"] = orjson["_via_settings"]
    # 2. 根据 试卷级图片 和 rec json, 拆图片 和新json
    tmplsit = {}
    for i1 in orjson["_via_img_metadata"]:
        # 2.1遍历文件夹对应的每一个图片
        if orjson["_via_img_metadata"][i1]["filename"] in orlist:
            tfname = os.path.join(oripath, orjson["_via_img_metadata"][i1]["filename"])
            im = Image.open(tfname)
            # image = np.array(im).astype(np.float)
            image = np.array(im).astype(np.int)
            # print(os.path.join(oripath, orjson["_via_img_metadata"][i1]["filename"]))
            # print(image.shape)
            tcount = 0
            tablelist = []
            latexlist = []
            normallist = []
            # 2.2 获取全局信息，保存分图
            for i2 in orjson["_via_img_metadata"][i1]["regions"]:
                if i2["region_attributes"]["markr"] == tablestr:
                    # 2.2.1 遍历每一个区域
                    tcount += 1
                    # 读取区域
                    (filepath, tempfilename) = os.path.split(tfname)
                    (filename, extension) = os.path.splitext(tempfilename)
                    newname = filename + "_" + str(tcount).zfill(4) + extension
                    # print(newname)
                    x1, x2, y1, y2 = getregion(i2["shape_attributes"])
                    # print(x1, x2, y1, y2)
                    # print(image[y1:y2, x1:x2, :].shape)
                    tmpjsonss = {"filename": newname, "size": 3 * (y2 - y1) * (x2 - x1)}
                    # 2.2.2 保存区域
                    skimage.io.imsave(os.path.join(newpath, newname), image[y1:y2, x1:x2, :])
                    # 2.2.3 写入新json
                    tablelist.append([tmpjsonss, [x1, x2, y1, y2]])
                if i2["region_attributes"]["markr"] == latexstr:
                    # 2.2 遍历每一个标签
                    latexlist.append(i2)
                if i2["region_attributes"]["markr"] == normalstr:
                    # 2.2 遍历每一个标签
                    normallist.append(i2)
    #         for i2 in tablelist:
    #             regjsonn = i2[0]
    #             tlist = []
    #             for i3 in latexlist:
    #                 x1, x2, y1, y2 = getregion(i3["shape_attributes"])
    #                 xm = (x1 + x2) / 2
    #                 ym = (y1 + y2) / 2
    #                 if i2[1][0] < xm and i2[1][1] > xm and i2[1][2] < ym and i2[1][3] > ym:
    #                     # 2.2 遍历每一个区域
    #                     ti3 = copy.deepcopy(i3)
    #                     ti3["shape_attributes"]["all_points_x"], ti3["shape_attributes"]["all_points_y"] = modxylist(
    #                         ti3["shape_attributes"]["all_points_x"], ti3["shape_attributes"]["all_points_y"], *i2[1])
    #                     tlist.append(ti3)
    #             for i3 in normallist:
    #                 x1, x2, y1, y2 = getregion(i3["shape_attributes"])
    #                 xm = (x1 + x2) / 2
    #                 ym = (y1 + y2) / 2
    #                 if i2[1][0] < xm and i2[1][1] > xm and i2[1][2] < ym and i2[1][3] > ym:
    #                     # 2.2 遍历每一个区域
    #                     ti3 = copy.deepcopy(i3)
    #                     ti3["shape_attributes"]["all_points_x"], ti3["shape_attributes"]["all_points_y"] = modxylist(
    #                         ti3["shape_attributes"]["all_points_x"], ti3["shape_attributes"]["all_points_y"], *i2[1])
    #                     tlist.append(ti3)
    #             regjsonn["regions"] = tlist
    #             tmplsit[regjsonn["filename"]] = regjsonn
    # newjson["_via_img_metadata"] = tmplsit
    # # 3. 按rec存子图
    # with open(newfile, "w") as f:
    #     json.dump(newjson, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
