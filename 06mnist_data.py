# coding=utf-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import struct
from array import array
import os


def main():
    baspath = os.path.join("D:\\", "Chrome下载", "mnist_dataset")
    with open(os.path.join(baspath, "train-labels.idx1-ubyte"), "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = array("B", f.read())
        print(magic, size, labels)

    with open(os.path.join(baspath, "t10k-images.idx3-ubyte"), "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        print(magic, size, rows, cols)
        image_data = array("B", f.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    for i, img in enumerate(images):
        if i < 72:
            plt.subplot(9, 8, i + 1)
            img = np.array(img)
            img = img.reshape(rows, cols)
            img = Image.fromarray(img)
            plt.imshow(img, cmap='gray')
            plt.axis("off")
        else:
            break


def chinese_ocr(pathn):
    # 1. 目录 解析文件
    fillist = os.listdir(pathn)
    contlist = []
    for onfile in fillist:
        with open(os.path.join(pathn, onfile), "r", encoding="utf-8") as f:
            diclist = f.readlines()
        contlist.append([".".join(onfile.split(".")[:-1]) + ".jpg", diclist])
    # 2. 字典文件获取
    dicfile = os.path.join("/home", "aa", "projectl", "Mask_RCNN", "densenet", "char_std_5990.txt")
    with open(dicfile, "r", encoding="utf-8") as f:
        diclist = f.readlines()
        diclist = [one.strip() for one in diclist]
    # 3. 修改每一个contlist
    print(len(diclist))
    for idn in range(len(contlist)):
        tindlist = []
        print(contlist[idn])
        try:
            for chars in contlist[idn][1][0]:
                try:
                    tind = diclist.index(chars)
                except Exception as e:
                    tind = len(diclist)
                    diclist.append(chars)
                tindlist.append(str(tind))
        except Exception as e:
            pass
        contlist[idn][1] = " ".join(tindlist)
        contlist[idn] = " ".join(contlist[idn])
    contlist = "\n".join(contlist)
    print(len(diclist))
    # print(diclist)
    # 4. 目录 解析文件
    with open(pathn + ".txt", "w", encoding="utf-8") as f:
        f.write(contlist)
    with open(dicfile, "w", encoding="utf-8") as f:
        diclist = "\n".join(diclist)
        f.write(diclist)


def num2char():
    pathn = os.path.join("C:\\", "project", "data", "ocr", "Synthetic_Chinese_String_Dataset")
    contdic = {}
    with open(os.path.join(pathn, "label.txt"), "r", encoding="utf-8") as f:
        diclist = f.readlines()
        diclist = [i1.strip() for i1 in diclist]
        contdic = {str(id1): i1 for id1, i1 in enumerate(diclist)}
    contdic["0"] = " "
    onfile = "data_test.txt"
    with open(os.path.join(pathn, onfile), "r", encoding="utf-8") as f:
        diclist = f.readlines()
        diclist = [i1.strip() for i1 in diclist]
        diclist = [i1.split(" ") for i1 in diclist]
    contlist = []
    for line in diclist:
        tmplist = []
        for id1, chars in enumerate(line):
            if id1 == 0:
                tmplist.append(chars)
            else:
                tmplist.append(contdic[chars])
        contlist.append(tmplist[0] + " " + "".join(tmplist[1:]))
    contlist = "\n".join(contlist)
    with open(os.path.join(pathn, "chi_" + onfile), "w", encoding="utf-8") as f:
        f.write(contlist)
    onfile = "data_train.txt"
    with open(os.path.join(pathn, onfile), "r", encoding="utf-8") as f:
        diclist = f.readlines()
        diclist = [i1.strip() for i1 in diclist]
        diclist = [i1.split(" ") for i1 in diclist]
    contlist = []
    for line in diclist:
        tmplist = []
        for id1, chars in enumerate(line):
            if id1 == 0:
                tmplist.append(chars)
            else:
                tmplist.append(contdic[chars])
        contlist.append(tmplist[0] + " " + "".join(tmplist[1:]))
    contlist = "\n".join(contlist)
    with open(os.path.join(pathn, "chi_" + onfile), "w", encoding="utf-8") as f:
        f.write(contlist)


if __name__ == '__main__':
    # main()
    num2char()
    exit()
    pathn = "HWDB2.0Train_label"
    print(pathn)
    chinese_ocr(pathn)
    pathn = "HWDB2.0Test_label"
    print(pathn)
    chinese_ocr(pathn)
    pathn = "HWDB2.1Train_label"
    print(pathn)
    chinese_ocr(pathn)
    pathn = "HWDB2.1Test_label"
    print(pathn)
    chinese_ocr(pathn)
    pathn = "HWDB2.2Train_label"
    print(pathn)
    chinese_ocr(pathn)
    pathn = "HWDB2.2Test_label"
    print(pathn)
    chinese_ocr(pathn)
