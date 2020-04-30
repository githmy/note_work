"""
右脑多媒体
"""
import os
import time
import cv2
import numpy as np
from sympy import *
import pytesseract
from PIL import Image
import csv
import re
import json


def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    img = cv2.warpAffine(image, M, (w, h))
    return img


def rotate_points(points, angle, cX, cY):
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0).astype(np.float16)
    a = M[:, :2]
    b = M[:, 2:]
    b = np.reshape(b, newshape=(1, 2))
    a = np.transpose(a)
    points = np.dot(points, a) + b
    points = points.astype(np.int)
    return points


def findangle(_image):
    # 用来寻找当前图片文本的旋转角度 在±90度之间
    # toWidth: 特征图大小：越小越快 但是效果会变差
    # minCenterDistance：每个连通区域坐上右下点的索引坐标与其质心的距离阈值 大于该阈值的区域被置0
    # angleThres：遍历角度 [-angleThres~angleThres]

    toWidth = _image.shape[1] // 2  # 500
    minCenterDistance = toWidth / 20  # 10
    angleThres = 45

    image = _image.copy()
    h, w = image.shape[0:2]
    if w > h:
        maskW = toWidth
        maskH = int(toWidth / w * h)
    else:
        maskH = toWidth
        maskW = int(toWidth / h * w)
    # 使用黑色填充图片区域
    swapImage = cv2.resize(image, (maskW, maskH))
    # grayImage = cv2.cvtColor(swapImage, cv2.COLOR_BGR2GRAY)
    grayImage = swapImage
    gaussianBlurImage = cv2.GaussianBlur(grayImage, (3, 3), 0, 0)
    histImage = cv2.equalizeHist(~gaussianBlurImage)
    binaryImage = cv2.adaptiveThreshold(histImage, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    # pointsNum: 遍历角度时计算的关键点数量 越多越慢 建议[5000,50000]之中
    pointsNum = np.sum(binaryImage != 0) // 2

    # # 使用最小外接矩形返回的角度作为旋转角度
    # # >>一步到位 不用遍历
    # # >>如果输入的图像切割不好 很容易受干扰返回0度
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dilated = cv2.dilate(binaryImage*255, element)
    # dilated = np.pad(dilated,((50,50),(50,50)),mode='constant')
    # cv2.imshow('dilated', dilated)
    # coords = np.column_stack(np.where(dilated > 0))
    # angle = cv2.minAreaRect(coords)
    # print(angle)

    # 使用连接组件寻找并删除边框线条
    # >>速度比霍夫变换快5~10倍 25ms左右
    # >>计算每个连通区域坐上右下点的索引坐标与其质心的距离，距离大的即为线条
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage, connectivity, cv2.CV_8U)
    labels = np.array(labels)
    maxnum = [(i, stats[i][-1], centroids[i]) for i in range(len(stats))]
    maxnum = sorted(maxnum, key=lambda s: s[1], reverse=True)
    if len(maxnum) <= 1:
        return 0
    for i, (label, count, centroid) in enumerate(maxnum[1:]):
        cood = np.array(np.where(labels == label))
        distance1 = np.linalg.norm(cood[:, 0] - centroid[::-1])
        distance2 = np.linalg.norm(cood[:, -1] - centroid[::-1])
        if distance1 > minCenterDistance or distance2 > minCenterDistance:
            binaryImage[labels == label] = 0
        else:
            break
    # cv2.imshow('after process', binaryImage * 255)

    minRotate = 0
    minCount = -1
    (cX, cY) = (maskW // 2, maskH // 2)
    points = np.column_stack(np.where(binaryImage > 0))[:pointsNum].astype(np.int16)
    for rotate in range(-angleThres, angleThres):
        rotatePoints = rotate_points(points, rotate, cX, cY)
        rotatePoints = np.clip(rotatePoints[:, 0], 0, maskH - 1)
        hist, bins = np.histogram(rotatePoints, maskH, [0, maskH])
        # 横向统计非零元素个数 越少则说明姿态越正
        zeroCount = np.sum(hist > toWidth / 50)
        if zeroCount <= minCount or minCount == -1:
            minCount = zeroCount
            minRotate = rotate

    # print("over: rotate = ", minRotate)
    return minRotate


def angle_handle(file):
    # done:
    cv_img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # print(cv_img)
    # print(cv_img.shape)
    mindic = {}
    t = time.time()
    for agl in range(-10, 10):
        img = cv_img.copy()
        img = rotate_bound(img, agl)
        # cv2.imshow('rotate', img)
        # t = time.time()
        angle = findangle(img)
        mindic[str(agl)] = angle
        # print(agl, angle, time.time() - t)
        img = rotate_bound(img, -angle)
        # cv2.imshow('after', img)
        # cv2.waitKey(200)
    finalang = int(sorted(mindic.items(), key=lambda x: abs(x[1]))[0][0])
    # print(finalang, time.time() - t)
    picnp = rotate_bound(cv_img, finalang)
    # print(picnp)
    # print(picnp.shape)
    # cv2.imshow('rotated', picnp)
    # cv2.waitKey(20000)
    return picnp


def table_handle(picnp):
    # todo:
    # 1. 识别出区块
    tmppicnp = picnp.copy()
    list2pics = []
    return list2pics


def table_handle_hard(picnp):
    # todo:
    # 1. 识别出区块
    tmppicnp = picnp.copy()
    # 二值化
    binary = cv2.adaptiveThreshold(~tmppicnp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    cv2.imshow("binary_picture", binary)  # 展示图片
    rows, cols = binary.shape
    # scale = 40
    scale = 80
    # 自适应获取核值 识别横线
    # 图像必须是二值化的. 核对应的原图像的所有像素值都是 1，那么中心元素就保持原来的像素值，否则就变为零（异或运算）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)

    dilated_col = cv2.dilate(eroded, kernel, iterations=1)
    cv2.imshow("excel_horizontal_line", dilated_col)
    # cv2.waitKey(0)
    # 识别竖线
    # scale = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_row = cv2.dilate(eroded, kernel, iterations=1)
    cv2.imshow("excel_vertical_line", dilated_row)
    # cv2.waitKey(0)
    # 标识交点
    bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
    cv2.imshow("excel_bitwise_and", bitwise_and)
    # cv2.waitKey(0)
    # 标识表格
    merge = cv2.add(dilated_col, dilated_row)
    cv2.imshow("entire_excel_contour", merge)
    # cv2.waitKey(0)
    # 两张图片进行减法运算，去掉表格框线
    merge2 = cv2.subtract(binary, merge)
    cv2.imshow("binary_sub_excel_rect", merge2)

    new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erode_image = cv2.morphologyEx(merge2, cv2.MORPH_OPEN, new_kernel)
    cv2.imshow('erode_image2', erode_image)
    merge3 = cv2.add(erode_image, bitwise_and)
    cv2.imshow('merge3', merge3)
    # cv2.waitKey(0)
    # 识别黑白图中的白色交叉点，将横纵坐标取出
    ys, xs = np.where(bitwise_and > 0)
    # 2. 区块分组 [step, label]
    #
    list2pics = []
    # 纵坐标
    y_point_arr = []
    # 横坐标
    x_point_arr = []
    # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
    # 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
    i = 0
    sort_x_point = np.sort(xs)
    for i in range(len(sort_x_point) - 1):
        if sort_x_point[i + 1] - sort_x_point[i] > 10:
            x_point_arr.append(sort_x_point[i])
        i = i + 1
    x_point_arr.append(sort_x_point[i])  # 要将最后一个点加入

    i = 0
    sort_y_point = np.sort(ys)
    # print(np.sort(ys))
    for i in range(len(sort_y_point) - 1):
        if (sort_y_point[i + 1] - sort_y_point[i] > 10):
            y_point_arr.append(sort_y_point[i])
        i = i + 1
    # 要将最后一个点加入
    y_point_arr.append(sort_y_point[i])
    print('y_point_arr', y_point_arr)
    print('x_point_arr', x_point_arr)
    # 循环y坐标，x坐标分割表格
    data = [[] for _ in range(len(y_point_arr))]
    for i in range(len(y_point_arr) - 1):
        for j in range(len(x_point_arr) - 1):
            # 在分割时，第一个参数为y坐标，第二个参数为x坐标
            cell = picnp[y_point_arr[i]:y_point_arr[i + 1], x_point_arr[j]:x_point_arr[j + 1]]
            cv2.imshow("sub_pic" + str(i) + "_" + str(j), cell)
            # # 读取文字，此为默认英文
            # # pytesseract.pytesseract.tesseract_cmd = 'E:/Tesseract-OCR/tesseract.exe'
            # text1 = pytesseract.image_to_string(cell, lang="chi_sim")
            #
            # # 去除特殊字符
            # text1 = re.findall(r'[^\*"/:?\\|<>″′‖ 〈\n]', text1, re.S)
            # text1 = "".join(text1)
            # print('单元格图片信息：' + text1)
            # data[i].append(text1)
            j = j + 1
        i = i + 1
    # cv2.waitKey(0)
    return list2pics


def texttype_handle(list2pics):
    # todo:
    list2dicpics = []
    return list2dicpics


def content_handle(list2dicpics):
    # todo:
    list2dicstrs = []
    return list2dicstrs


def answer_handle(answer, list2dicstrs):
    # todo:
    points = []
    # [n, 3]
    list2dicstrs = [
        ["5", "6", [{"type": "text", "txt": "解"}, {"type": "latex", "txt": "\\frac {3}{5}"},
                    {"type": "text", "txt": "得"}]],
        ["5", "7", [{"type": "latex", "txt": "=0.8"}]],
    ]
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    aa = solve(Eq(y, x + x ** 2), x)
    print(aa)
    s = "1 + 2"
    print(eval(s))
    # 1. 循环每一步
    for i1 in list2dicstrs:
        # 2. 解析每一步
        print(i1)
    return points


def print_solve(file):
    # todo: 简单识别，数据库对比提取。可能用不到。
    answer = ""
    return answer


def hand_solve(file):
    # todo:
    # 1. 角度纠正 输入文件名 返回纠正图片
    picnp = angle_handle(file)
    # 2. 表格划分 输入图片内容 返回二维数组的图片集 [n,3] pics
    list2pics = table_handle(picnp)
    # 3. 文字类型区分 输入二维数组的图片集 返回二维数组的字典图片集
    list2dicpics = texttype_handle(list2pics)
    # 4. 内容识别 输入二维数组的字典图片集 返回二维数组的字典字符串
    list2dicstrs = content_handle(list2dicpics)
    return list2dicstrs


def main():
    # todo:
    printfile = os.path.join("C:\\Users\Smile\Desktop", "tmp", "demo1.png")
    handfile = os.path.join("C:\\Users\Smile\Desktop", "tmp", "demo1.png")
    # 1. 问题解析
    answer = print_solve(printfile)
    # 2. 答案解析
    list2dicstrs = hand_solve(handfile)
    # 3. 答案对比
    res = answer_handle(answer, list2dicstrs)
    print(res)
    return res


if __name__ == '__main__':
    main()
