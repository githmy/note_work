# codinddg:ut5f-8
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
    # 1、 文件读入
    with open(os.path.join(project_path, "{}.net".format(filehead)), 'rb') as f:
        data = f.read()
    ftype = chardet.detect(data)["encoding"]
    modelhead = ["^D", "^Q", "^T", "^M"]
    tcadhead = ["^TCAD_"]
    twohead = ["^D", "^L", "^R", "^C", "^G", "^E", "^F", "^H", "^I", "^V"]
    threehead = ["^T"]
    fourhead = ["^Q", "^M"]
    headlist = ["^\.model", "^\.lib", ".*TCAD_*", "^\.tran", "^\.end"]
    with codecs.open(os.path.join(project_path, "{}.net".format(filehead)), "r", encoding=ftype) as inhand:
        # 1. 读入
        outlist = []
        readsig = True
        while readsig:
            incont = inhand.readline()
            for i1 in headlist + twohead + threehead + fourhead:
                rem = re.search(i1, incont, re.IGNORECASE)
                if rem:
                    if i1 == ".*TCAD_*":
                        incont = incont[rem.end() - len("TCAD_"):]
                    elif i1 == "^\.end":
                        readsig = False
                    incont = incont.replace("µ", "u")
                    outlist.append(incont)
                    break
    # 2. 分类存储
    ordilist = []
    modellist = []
    for line in outlist:
        if re.search("^\.model ", line, re.IGNORECASE):
            modellist.append(line)
            continue
        if re.search("^\.lib ", line, re.IGNORECASE):
            line = line[len(".lib"):].strip()
            line = line.encode("latin1").decode("gbk")
            with open(line, 'rb') as f:
                data = f.read()
            ftype = chardet.detect(data)["encoding"]
            with codecs.open(line, "r", encoding=ftype) as inhand:
                # modellist += [i2 for i2 in inhand.readlines() if re.search("^\.model ", i2, re.IGNORECASE)]
                tliblist = []
                for i2 in inhand.readlines():
                    if re.search("^\.model ", i2, re.IGNORECASE):
                        modellist.append("".join(tliblist))
                        tliblist = [i2]
                    elif re.search("^\+", i2):
                        tliblist.append(i2)
                modellist.append("".join(tliblist))
            continue
        ordilist.append(line)
    ordilist = [i1.strip() for i1 in ordilist]
    modellist = [i1 for i1 in modellist if i1 != ""]
    modellist = [i1.strip() for i1 in modellist]
    # 3. 获取节点，再编号
    nodelist = {}
    for command in ordilist:
        if re.search(tcadhead[0], command, re.IGNORECASE):
            tlist = command.split()
            tlist = tlist[1:-1]
            for i3 in tlist:
                nodelist[i3] = None
            continue
        for i2 in twohead:
            # 只取第二，第三个
            if re.search(i2, command, re.IGNORECASE):
                tlist = command.split()
                nodelist[tlist[1]] = None
                nodelist[tlist[2]] = None
                break
        for i2 in threehead:
            if re.search(i2, command, re.IGNORECASE):
                tlist = command.split()
                nodelist[tlist[1]] = None
                nodelist[tlist[2]] = None
                try:
                    nodelist[tlist[3]] = None
                except Exception as e:
                    pass
                break
        for i2 in fourhead:
            # 只取 第二, 第三, 第四, 第五
            if re.search(i2, command, re.IGNORECASE):
                tlist = command.split()
                nodelist[tlist[1]] = None
                nodelist[tlist[2]] = None
                nodelist[tlist[3]] = None
                nodelist[tlist[4]] = None
                break
    del nodelist["0"]
    nodecounter = 0
    for key in nodelist.keys():
        nodecounter += 1
        nodelist[key] = str(nodecounter)
    nodelist["0"] = "0"
    for id1, command in enumerate(ordilist):
        if re.search(tcadhead[0], command, re.IGNORECASE):
            tlist = command.split()
            dlenth = len(set(tlist[1:-1]))
            tlist[1:dlenth + 1] = [nodelist[i3] for i3 in tlist[1:dlenth + 1]]
            tlist = tlist[0:dlenth + 1] + [tlist[-1]]
            ordilist[id1] = " ".join(tlist)
            continue
        for i2 in twohead:
            # 只取第二，第三个
            if re.search(i2, command, re.IGNORECASE):
                tlist = command.split()
                tlist[1] = nodelist[tlist[1]]
                tlist[2] = nodelist[tlist[2]]
                ordilist[id1] = " ".join(tlist)
                break
        for i2 in threehead:
            # 只取第二，第三个
            if re.search(i2, command, re.IGNORECASE):
                tlist = command.split()
                tlist[1] = nodelist[tlist[1]]
                tlist[2] = nodelist[tlist[2]]
                try:
                    tlist[3] = nodelist[tlist[3]]
                except Exception as e:
                    pass
                ordilist[id1] = " ".join(tlist)
                break
        for i2 in fourhead:
            # 只取第二，第三个
            if re.search(i2, command, re.IGNORECASE):
                tlist = command.split()
                tlist[1] = nodelist[tlist[1]]
                tlist[2] = nodelist[tlist[2]]
                tlist[3] = nodelist[tlist[1]]
                tlist[4] = nodelist[tlist[2]]
                ordilist[id1] = " ".join(tlist)
                break
    # 4. 只保留有效的库文件
    modelkey = []
    for command in ordilist:
        if re.search(tcadhead[0], command, re.IGNORECASE):
            continue
        for i2 in modelhead:
            if re.search(i2, command, re.IGNORECASE):
                tlist = command.split()
                if i2 in twohead:
                    modelkey.append(tlist[3])
                elif i2 in threehead:
                    # 加错了没关系，找不到映射不会输出
                    try:
                        _ = int(tlist[4])
                        modelkey.append(tlist[4])
                    except Exception as e:
                        modelkey.append(tlist[3])
                elif i2 in fourhead:
                    modelkey.append(tlist[5])
                break
            pass
    modelremain = []
    for i1 in modellist:
        if i1.split()[1] in modelkey:
            modelremain.append(i1)
    # print(ordilist)
    # print(modellist)
    # print(modelkey)
    # print(modelremain)
    ordilist = ordilist[0:-1] + modelremain + [ordilist[-1]]
    # 5. 写出
    with codecs.open(os.path.join(project_path, "{}.cir".format(filehead)), "w", encoding="utf8") as f:
        for i1 in ordilist:
            f.write(i1 + "\r\n")


if __name__ == '__main__':
    main()
