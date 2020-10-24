# coding:utf-8
import os
import pandas as pd
from sklearn.cluster import KMeans
import time
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc_special
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from collections import OrderedDict
from sklearn.manifold import TSNE
import math
import shutil
import re
import codecs


def judge_in(infile):
    inhand = codecs.open(infile, "r", "utf8")
    incont = inhand.readlines()
    stdcounter = 0
    for id1, line in enumerate(incont):
        line = line.strip()
        incont[id1] = line.split("$")[0]
    for id1 in range(len(incont) - 2, -1, -1):
        incont[id1] = incont[id1].strip()
        if re.search(r"&&$", incont[id1]):
            incont[id1] = incont[id1].replace("&&", incont[id1 + 1])
            incont[id1 + 1] = ""
    newincont = []
    for line in incont:
        if line != "":
            newincont.append(line)
    incont = newincont
    for line in incont:
        line = line.strip()
        if stdcounter > 0 and re.search(r"^scan ", line):
            if "print_step" in line:
                line = ' '.join(line.split())
                line = line.replace("= ", "=")
                line = line.replace(" =", "=")
                line = [sp.replace("print_step=", "") for sp in line.split() if re.search("print_step", sp)][0]
                stdcounter += int(line)
            else:
                stdcounter += 1
        if re.search(r"^equilibrium", line):
            stdcounter += 1
    # print(stdcounter)
    outhand = codecs.open(infile + ".log", "r", "utf8")
    outcont = outhand.readlines()
    print(outcont)
    finalstr = ""
    for line in outcont:
        line = line.strip()
        if re.search(r"\.out_", line):
            finalstr = line
    print(finalstr)
    finalstr = finalstr.split("out_")[-1]
    finalstr = finalstr.lstrip("0")
    if int(finalstr) >= stdcounter:
        return f"keytest ok {finalstr}>={stdcounter}"
    else:
        return f"keytest warning {finalstr}<{stdcounter}"


def deal_file(csuprem_path, apsys_path, projectpath, filename):
    # 1. 执行文件
    filesuffix = filename.split(".")[-1]
    starttime = time.time()
    os.chdir(projectpath)
    if filesuffix == "in":
        commandstr = f'del *.log && del *.str'
        os.system(commandstr)
        tmpexe = os.path.join(csuprem_path, 'csuprem')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
        print(f"usetime csuprem: {(time.time()-starttime)/60}min")
    elif filesuffix == "layer":
        tmpexe = os.path.join(apsys_path, 'layer')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
    elif filesuffix == "gain":
        tmpexe = os.path.join(apsys_path, 'apsys')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
    elif filesuffix == "geo":
        tmpexe = os.path.join(apsys_path, 'apsys')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
    elif filesuffix == "mplt":
        tmpexe = os.path.join(apsys_path, 'apsys')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
    elif filesuffix == "sol":
        tmpexe = os.path.join(apsys_path, 'apsys')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
        print(f"usetime apsys: {(time.time()-starttime)/60}min")
        keyoutstr = judge_in(filename)
        return keyoutstr
        # 2. 判断
    elif filesuffix == "plt":
        tmpexe = os.path.join(apsys_path, 'apsys')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)


def test_batch(csuprem_path, apsys_path, example_path):
    # 1. 便利目录，找到 in layer sol 文件
    noiterdir = []
    testlist = []
    key_suffix = ["\.layer$", "\.gain$", "\.geo$", "\.mplt$", "\.in$", "\.sol$", "\.plt$"]
    for root, dirs, files in os.walk(example_path):
        # 1.1 跳过自动生成目录
        passig = 0
        for noiter in noiterdir:
            if noiter in root:
                passig = 1
                break
        if passig == 1:
            continue
        # 1.2 预清除输出文件
        os.chdir(root)
        commandstr = f'del *.info && del *.ac && del *.tmp && del *.ps && del *.out* && del *.std* && del *.zp* && del *.ar* && del *.msg && del *.log && del *.mon* && del fort.* && del *.qws && del *.rta* && del *.rti && del *.rtm* && del *.rto* && del *.sho && del *.sho*'
        os.system(commandstr)
        print(f"path {root}")
        # 1.3 查找关键文件
        for key in key_suffix:
            # 1.4 按 顺序一次处理 相同类型的文件
            fils = [file for file in files if re.search(key, file)]
            if key == "\.layer$":
                pass
            if key == "\.in$":
                fils = [fil for fil in fils if not re.search("^geo", fil)]
            if key == "\.sol$":
                fils = [fil for fil in fils if not re.search("^material_[2..3]d", fil)]
            if key == "\.plt$":
                pass
            if len(fils) > 0:
                noiterdir += [os.path.join(root, dir) for dir in dirs]
                print(fils)
                for fil in fils:
                    keyoutstr = deal_file(csuprem_path, apsys_path, root, fil)
                    testlist.append(keyoutstr)


if __name__ == '__main__':
    infile = "C:\project\EAM\CQW_modulator\ingaalas.sol"
    judge_in(infile)
    raise 666
    csuprem_path = os.path.join("C:\\", "project", "Csuprem", "Bin")
    apsys_path = os.path.join("C:\\", "project", "crosslig_apsys", "apsys")
    # example_path = os.path.join("c:\\", "project", "linux_core", "apsys_examples")
    example_path = os.path.join("c:\\", "project", "EAM")
    test_batch(csuprem_path, apsys_path, example_path)
