# coding:utf-8
import os
import time
import chardet
import copy
import json
import math
import shutil
import re
import codecs
import math
import platform
import datetime
import logging
import logging.handlers


def test_batch(csuprem_path, apsys_path, example_path, purpose):
    # 1. 便利目录，找到 in layer sol 文件
    noiterdir = []
    testlist = []
    # key_suffix = ["\.in$", "\.layer$", "\.gain$", "\.geo$", "\.mplt$", "\.sol$", "\.plt$"]
    key_suffix = ["\.layer$", "\.cut$", "\.in$", "\.geo$", "\.sol$", "\.plt$"]
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
        logger.info(f"path {root}")
        # 1.3 查找关键文件
        for key in key_suffix:
            # 1.4 按 顺序一次处理 相同类型的文件
            files = os.listdir(root)
            fils = [file for file in files if re.search(key, file)]
            # print("fils", fils)
            if key == "\.layer$":
                pass
            if key == "\.cut$":
                pass
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
                logger.info(fils)
                for fil in fils:
                    try:
                        keyoutstr = deal_file(csuprem_path, apsys_path, root, fil, purpose)
                        testlist.append(keyoutstr)
                    except Exception as e:
                        logger.info(e)
                    time.sleep(2)


def main():
    project_path = os.path.join("G:\\", "Program Files (x86)", "LTspiceIV 汉化版", "examples", "ttest")
    filehead = "draft1"
    with open(os.path.join(project_path, "{}.net".format(filehead)), 'rb') as f:
        data = f.read()
    ftype = chardet.detect(data)["encoding"]
    headlist = ["^V", "^I", "^R", "^L", "^C", "^D", "^\.model", "^\.lib", ".*TCAD_*", "^\.tran", "^\.end"]
    with codecs.open(os.path.join(project_path, "{}.net".format(filehead)), "r", encoding=ftype) as inhand:
    # with codecs.open(os.path.join(project_path, "{}.net".format(filehead)), "r", encoding=None) as inhand:
        # 1. 读入
        outlist = []
        readsig = True
        while readsig:
            incont = inhand.readline()
            for i1 in headlist:
                rem = re.search(i1, incont, re.IGNORECASE)
                if rem:
                    if i1 == ".*TCAD_*":
                        incont = incont[rem.end() - len("TCAD_"):]
                    elif i1 == "^\.end":
                        readsig = False
                    incont = incont.replace("µ", "u")
                    outlist.append(incont)
                    break
    # 2. 获取节点，再编号
    nodelist = {}
    liblist = []
    for line in outlist:
        if re.search("^\.lib ", line, re.IGNORECASE):
            liblist.append(line)
            line = line[len(".lib"):].strip()
            print(line)
            line=line.encode()
            print(line)
            line=line.decode("GB18030")
            print(line)
            with open(line, 'rb') as f:
                data = f.read()
            ftype = chardet.detect(data)["encoding"]
            with codecs.open(line, "r", encoding=ftype) as inhand:
                print(inhand.readlines())
            continue
        tlist = line.split()
        print(tlist)


if __name__ == '__main__':
    main()
