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
import math
import platform
import datetime


def judge_in(infile):
    inhand = codecs.open(infile, "r", "utf8")
    incont = inhand.readlines()
    finalstr = ""
    for id1, line in enumerate(incont):
        line = line.strip()
        incont[id1] = line.split("#")[0].strip()
        if re.search("^export ", incont[id1]):
            finalstr = incont[id1]
    finalstr = ' '.join(finalstr.split())
    finalstr = finalstr.replace("= ", "=")
    finalstr = finalstr.replace(" =", "=")
    finalstr = [sp.replace("outf=", "") for sp in finalstr.split() if re.search("outf=", sp)][0]
    if finalstr in os.listdir("."):
        return f"keytest ok {finalstr} exist"
    else:
        return f"keytest warning {finalstr} not exist"


def judge_sol(infile):
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
                tstep = \
                    [sp.replace("print_step=", "").replace("D", "e") for sp in line.split() if
                     re.search("print_step", sp)][
                        0]
                tvalu = \
                    [sp.replace("value_to=", "").replace("D", "e") for sp in line.split() if re.search("value_to", sp)][
                        0]
                stdcounter += abs(math.ceil(float(tvalu) / float(tstep)))
                # print(tvalu, tstep, float(tvalu), float(tstep), abs(math.ceil(float(tvalu) / float(tstep))))
            else:
                stdcounter += 1
        if re.search(r"^equilibrium", line):
            stdcounter += 1
    # print(stdcounter)
    outhand = codecs.open(infile + ".log", "r", "utf8")
    outcont = outhand.readlines()
    finalstr = ""
    for line in outcont:
        if re.search(r"\.out_", line):
            finalstr = line
    finalstr = finalstr.split("out_")[-1]
    finalstr = finalstr.lstrip("0")
    finalstr = int(finalstr)
    if finalstr == stdcounter:
        return f"keytest ok purpose {stdcounter} real {finalstr}"
    else:
        return f"keytest warning purpose {stdcounter} real {finalstr}"


def deal_file(csuprem_path, apsys_path, projectpath, filename):
    # 1. 执行文件
    splitlist = filename.split(".")
    filesuffix = splitlist[-1]
    starttime = time.time()
    os.chdir(projectpath)
    if filesuffix == "in":
        tmpexe = os.path.join(csuprem_path, 'csuprem')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
        print(f"usetime csuprem: {(time.time()-starttime)/60}min")
        keyoutstr = judge_in(filename)
        print(keyoutstr)
        return keyoutstr
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
        tmpexe = os.path.join(apsys_path, 'geometry')
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
        keyoutstr = judge_sol(filename)
        print(keyoutstr)
        return keyoutstr
        # 2. 判断
    elif filesuffix == "plt":
        tmpexe = os.path.join(apsys_path, 'apsys')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        print(commandstr)
        os.system(commandstr)
        # 判断gnuplot
        if (platform.system() == 'Windows'):
            tmpexe = os.path.join(apsys_path, 'gnuplot')
            commandstr = f'"{tmpexe}" junkg.tmp'
        elif (platform.system() == 'Linux'):
            tmpexe = 'gnuplot'
            commandstr = f'"{tmpexe}" junkg.tmp'
        else:
            print('其他系统')
        print(commandstr)
        os.system(commandstr)
        time.sleep(2)
        filhead = ".".join(splitlist[:-1])
        os.rename("output.ps", f"{filhead}.ps")
        time.sleep(2)


def test_batch(csuprem_path, apsys_path, example_path):
    # 1. 便利目录，找到 in layer sol 文件
    noiterdir = []
    testlist = []
    key_suffix = ["\.in$", "\.layer$", "\.gain$", "\.geo$", "\.mplt$", "\.sol$", "\.plt$"]
    rmlist = ["\.info$", "\.ac$", "\.tmp$", "\.ps$", "\.out", "\.std", "\.str$", "\.zp", "\.ar", "\.msg$", "\.log$",
              "\.mon", "^fort\.", "\.qws$", "\.rta", "\.rti$", "\.rtm", "\.rto", "\.sho$", "\.sho"]
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
        delfiles = []
        for key in rmlist:
            delfiles += [dfile for dfile in files if re.search(key, dfile)]
        for file in delfiles:
            os.remove(os.path.join(root, file))
        print(f"path {root}")
        # 1.3 查找关键文件
        for key in key_suffix:
            # 1.4 按 顺序一次处理 相同类型的文件
            files = os.listdir(root)
            fils = [file for file in files if re.search(key, file)]
            # print("fils", fils)
            if key == "\.in$":
                fils = [fil for fil in fils if not re.search("^geo", fil)]
                fils = [fil for fil in fils if fil not in ["csuprem_template.in", "temp.in"]]
                dffils = [fil for fil in fils if re.search("\.aps$", fil)]
                # print("del aps", dffils)
                dffils += [fil for fil in fils if re.search("\.log$", fil)]
                # print("del log", dffils)
                dffils += [fil for fil in fils if re.search("\.str$", fil)]
                # print("del str", dffils)
                for fil in dffils:
                    os.remove(os.path.join(root, fil))
            if key == "\.layer$":
                pass
            if key == "\.sol$":
                fils = [fil for fil in fils if not re.search("^material_[2..3]d", fil)]
                fils = [fil for fil in fils if not re.search("^contact_[2..3]d", fil)]
                fils = [fil for fil in fils if not re.search("^main_[2..3]d", fil)]
                dffils = [fil for fil in fils if re.search("\.std_", fil)]
                # print("del str", dffils)
                for fil in dffils:
                    os.remove(os.path.join(root, fil))
            if key == "\.plt$":
                pass
            if len(fils) > 0:
                noiterdir += [os.path.join(root, dir) for dir in dirs]
                print(fils)
                for fil in fils:
                    try:
                        keyoutstr = deal_file(csuprem_path, apsys_path, root, fil)
                        testlist.append(keyoutstr)
                    except Exception as e:
                        print(e)
                    time.sleep(2)


def main():
    if (platform.system() == 'Windows'):
        print('Windows系统')
        rootpath = "C:\\"
        csuprem_path = os.path.join(rootpath, "project", "crosslig_csuprem_ForLan_2020-12-07", "Bin")
        apsys_path = os.path.join(rootpath, "project", "crosslig_apsys_tmp", "apsys")
    elif (platform.system() == 'Linux'):
        print('Linux系统')
        rootpath = os.path.join("/", "home", "abc")
        csuprem_path = os.path.join("/", "opt", "crosslight", "csuprem-2020", "bin")
        # apsys_path = os.path.join("/", "opt", "crosslight", "apsys-2020", "bin")
        apsys_path = os.path.join("/", "opt", "crosslight", "apsys-2021", "bin")
        # csuprem_path = os.path.join("/", "opt", "empyrean", "aresps-2020", "bin")
        # apsys_path = os.path.join("/", "opt", "empyrean", "aresds-2020", "bin")
    else:
        print('其他')
        rootpath = "C:\\"
    print(csuprem_path)
    print(apsys_path)
    # example_path = os.path.join(rootpath, "project", "test_all")
    example_path = os.path.join(rootpath, "project", "simuproject", "gs", "All_Examples_2020.11.27", "csuprem_examples")
    print(example_path)
    print(datetime.datetime.now())
    test_batch(csuprem_path, apsys_path, example_path)
    print("all finished!")


if __name__ == '__main__':
    'cat this_log | grep "^keytest warning"'
    main()
    exit()
