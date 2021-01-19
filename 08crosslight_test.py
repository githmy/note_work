# coding:utf-8
import os
import time
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

datalogfile = 'crosslight_test.log'
logger = logging.getLogger('logger_out')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(datalogfile)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


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


def deal_file(csuprem_path, apsys_path, projectpath, filename, purpose):
    # 1. 执行文件
    splitlist = filename.split(".")
    filesuffix = splitlist[-1]
    starttime = time.time()
    os.chdir(projectpath)
    if filesuffix == "in":
        tmpexe = os.path.join(csuprem_path, 'csuprem')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
        logger.info(f"usetime csuprem: {(time.time()-starttime)/60}min")
        keyoutstr = judge_in(filename)
        logger.info(keyoutstr)
        return keyoutstr
    elif filesuffix == "cut":
        tmpexe = os.path.join(csuprem_path, 'MaskEditor', 'generate_mask_nognuplot')
        commandstr = f'"{tmpexe}" {filename} < input.txt > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
    elif filesuffix == "layer":
        tmpexe = os.path.join(apsys_path, 'layer')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
    elif filesuffix == "gain":
        tmpexe = os.path.join(apsys_path, purpose)
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
    elif filesuffix == "geo":
        tmpexe = os.path.join(apsys_path, 'geometry')
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
    elif filesuffix == "mplt":
        tmpexe = os.path.join(apsys_path, purpose)
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
    elif filesuffix == "sol":
        tmpexe = os.path.join(apsys_path, purpose)
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
        logger.info(f"usetime {purpose}: {(time.time()-starttime)/60}min")
        keyoutstr = judge_sol(filename)
        logger.info(keyoutstr)
        return keyoutstr
        # 2. 判断
    elif filesuffix == "plt":
        tmpexe = os.path.join(apsys_path, purpose)
        commandstr = f'"{tmpexe}" {filename} > {filename}.log'
        logger.info(commandstr)
        os.system(commandstr)
        # 判断gnuplot
        if (platform.system() == 'Windows'):
            tmpexe = os.path.join(apsys_path, 'gnuplot')
            commandstr = f'"{tmpexe}" junkg.tmp'
        elif (platform.system() == 'Linux'):
            tmpexe = 'gnuplot'
            commandstr = f'"{tmpexe}" junkg.tmp'
        else:
            logger.info('其他系统')
        logger.info(commandstr)
        os.system(commandstr)
        time.sleep(2)
        filhead = ".".join(splitlist[:-1])
        os.rename("output.ps", f"{filhead}.ps")
        time.sleep(2)


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
    purpose="apsys"
    # purpose = "pics3d"
    if (platform.system() == 'Windows'):
        logger.info('Windows系统')
        rootpath = "E:\\"
        # csuprem_path = os.path.join(rootpath, "project", "crosslig_csuprem_ForLan_2020-12-07", "Bin")
        csuprem_path = os.path.join(rootpath, "project", "versions_win", "crosslig_csuprem_ForLan_2020-12-07", "Bin")
        # pic3d_apsys_path = os.path.join(rootpath, "project", "versions_win", "crosslig_pics3d", "pics3d")
        pic3d_apsys_path = os.path.join(rootpath, "1_2df", "2_3dd", "crosslig_apsys", "apsys")
    elif (platform.system() == 'Linux'):
        logger.info('Linux系统')
        rootpath = os.path.join("/", "home", "abc")
        csuprem_path = os.path.join("/", "opt", "crosslight", "csuprem-2020", "bin")
        # apsys_path = os.path.join("/", "opt", "crosslight", "apsys-2020", "bin")
        pic3d_apsys_path = os.path.join("/", "opt", "crosslight", "apsys-2021", "bin")
        # csuprem_path = os.path.join("/", "opt", "empyrean", "aresps-2020", "bin")
        # apsys_path = os.path.join("/", "opt", "empyrean", "aresds-2020", "bin")
    else:
        logger.info('其他')
        rootpath = "C:\\"
    logger.info(csuprem_path)
    logger.info(pic3d_apsys_path)
    logger.info(purpose)
    # example_path = os.path.join(rootpath, "project", "test_all")
    example_path = os.path.join(rootpath, "project", "test_all", "pics3d_examples")
    logger.info(example_path)
    logger.info(datetime.datetime.now())
    test_batch(csuprem_path, pic3d_apsys_path, example_path, purpose)
    logger.info("all finished!")


if __name__ == '__main__':
    'cat this_log | grep "^keytest warning"'
    main()
    exit()
