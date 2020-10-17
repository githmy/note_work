# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from utils.request_t import ShishuoApi
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
import json
import csv
import numpy as np
import pandas as pd
import logging
import os
import codecs


# TXTPATH = "./txt"
# reslogfile = os.path.join(TXTPATH, 'result_no_other.log')
# if os._exists(reslogfile):
#     os.remove(reslogfile)
# # 创建一个logger
# logger1 = logging.getLogger('logger out')
# logger1.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
# fh = logging.FileHandler(reslogfile)
# # 再创建一个handler，用于输出到控制台
# ch = logging.StreamHandler()
#
# # 定义handler的输出格式formatter
# # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(message)s')
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)
#
# # logger1.addFilter(filter)
# logger1.addHandler(fh)
# logger1.addHandler(ch)


def txt2json(txtfin, jsonfout):
    my_matrix = np.loadtxt(open(txtfin, "rt"), dtype=np.str, delimiter=",", skiprows=0)
    # print(my_matrix)
    mapsingle = {'rasa_nlu_data': {'common_examples': []}}
    for i in my_matrix:
        mapsingle['rasa_nlu_data']['common_examples'].append({
            "text": i,
            "intent": "stock",
            "entities": []
        })
    with open(jsonfout, "w") as f:
        json.dump(mapsingle, f, ensure_ascii=False)
    return True


def csvddupli(csvin, csvout):
    df1 = pd.read_csv(csvin, header=0, encoding="utf8", dtype=str)
    df1.drop_duplicates(['text'], inplace=True)
    df1.sort_values(['intent'], inplace=True)
    df1.to_csv(csvout)


def csv21json(csvfin):
    df1 = pd.read_csv(csvfin, header=0, encoding="utf8", dtype=str)
    arrayobjs = {}
    for i in df1.index:
        ttobj = df1.iloc[i, :]
        if not arrayobjs.has_key(ttobj["intent"]):
            arrayobjs.__setitem__(ttobj["intent"], [])
        arrayobjs[ttobj["intent"]].append({
            "text": ttobj["text"],
            "intent": ttobj["intent"],
            "entities": []
        })
    listobj = []
    for i in arrayobjs:
        listobj.extend(arrayobjs[i])
    resobjs = {'rasa_nlu_data': {'common_examples': listobj}}
    with open(os.path.join(TXTPATH, "res") + ".json", "w") as f:
        json.dump(resobjs, f, ensure_ascii=False)


def csv2json(csvfin):
    df1 = pd.read_csv(csvfin, header=0, encoding="utf8", dtype=str)
    arrayobjs = {}
    for i in df1.index:
        ttobj = df1.iloc[i, :]
        if not arrayobjs.has_key(ttobj["intent"]):
            arrayobjs.__setitem__(ttobj["intent"], {'rasa_nlu_data': {'common_examples': []}})
        arrayobjs[ttobj["intent"]]['rasa_nlu_data']['common_examples'].append({
            "text": ttobj["text"],
            "intent": ttobj["intent"],
            "entities": []
        })
    for i in arrayobjs:
        with open(os.path.join(TXTPATH, str(i)) + ".json", "w") as f:
            json.dump(arrayobjs[i], f, ensure_ascii=False)

    # arrayobjs = {}
    # with open(csvfin, "rU") as csvfile:
    #     res = csv.reader(csvfile)
    #     for i in res:
    #         if not arrayobjs.has_key(i[1]):
    #             arrayobjs.__setitem__(i[1], {'rasa_nlu_data': {'common_examples': []}})
    #         arrayobjs[i[1]]['rasa_nlu_data']['common_examples'].append({
    #             "text": i[0],
    #             "intent": i[1],
    #             "entities": []
    #         })
    # for i in arrayobjs:
    #     with open(os.path.join(TXTPATH, str(i)) + ".json", "w") as f:
    #         json.dump(arrayobjs[i], f, ensure_ascii=False)
    return True


def log2csv(logjson, csvout):
    # 1. 读文件
    with open(logjson) as json_file:
        setting = json.loads(json_file.read())
        # print(setting)
        arrytxt = []
        map_txtmodel = {}
        # mapmores = {}
        mapintent = {}
        mapsingle = {'rasa_nlu_data': {'common_examples': []}}
        # df = pd.DataFrame([[5, 6, 7], [7, 8, 9]], columns=list(['text', 'model', 'intent', 'value']))
        # 2. 联合去重 text model， 最后一个intent 覆盖之前的结果
        for i in setting:
            # i['entities'] = []
            kkey = i['model'] + i['user_input']['text']
            vvalue = [j['value'] for j in i['user_input']['entities']]
            strvalue = "#".join(vvalue)
            if not map_txtmodel.has_key(kkey):
                map_txtmodel.__setitem__(kkey, [i['user_input']['text'], i['model'], i['user_input']['intent']['name'],
                                                strvalue])
            else:
                map_txtmodel[kkey] = [i['user_input']['text'], i['model'], i['user_input']['intent']['name'], strvalue]
        # 3. 写入csv文件
        with open(csvout, "wb") as csvfile:
            # fileheader = ['text', 'model', 'intent', 'value']
            # dict_writer = csv.DictWriter(csvfile, fileheader)
            # dict_writer.writeheader()
            # resultlist = [{'合计': '合计', '国有': 110, '集体': 112},
            #               {'合计': '国有', '国有': 50, '集体': 61},
            #               {'合计': '合计', '国有': 50, '集体': 40},
            #               {'合计': '集体', '国有': 15, '集体': 25}]
            # dict_writer.writerows(resultlist)

            writer = csv.writer(csvfile)
            # 先写入columns_name
            writer.writerow(['text', 'model', 'intent', 'value'])
            # 写入多行用writerows
            for i1 in map_txtmodel:
                writer.writerow(map_txtmodel[i1])
                # writer.writerows([map_txtmodel[i1]])
    return True


def json2csv(rootjson, csvout):
    # 1. 读文件
    with open(rootjson) as json_file:
        setting = json.loads(json_file.read())
        setts = setting['rasa_nlu_data']['common_examples']
        # 3. 写入csv文件
        with open(csvout, "w") as csvfile:
            writer = csv.writer(csvfile)
            # 先写入columns_name
            writer.writerow(['text', 'intent', 'dish'])
            # 写入多行用writerows
            for i in setts:
                tmp = ["dish" for i2 in i['entities'] if i2['entity'] == "dish"]
                try:
                    writer.writerows([[i['text'], i['intent'], tmp[0]]])
                except Exception as e:
                    writer.writerows([[i['text'], i['intent']], ""])
    return True


def drop_duplicate_json(jsonfin, jsonfout):
    # 1. 读文件
    with open(jsonfin) as json_file:
        setting = json.loads(json_file.read())
        # print(setting)
        maptxt = {}
        mapintent = {}
        mapsingle = {'rasa_nlu_data': {'common_examples': []}}
        mapsingleorder = {'rasa_nlu_data': {'common_examples': []}}
        # 1. 取唯一
        for i in setting['rasa_nlu_data']['common_examples']:
            # i['entities'] = []
            # if not i['text'] in maptxt:
            if not maptxt.has_key(i['text']):
                maptxt.__setitem__(i['text'], 0)
            maptxt[i['text']] = i
        # 2. 唯一累加格式化
        for i in maptxt:
            mapsingle['rasa_nlu_data']['common_examples'].append(maptxt[i])
        # # 3. 唯一累加写入json文件
        # with open(jsonfout, "w") as f:
        #     json.dump(mapsingle, f, ensure_ascii=False)
        # 4. intent划分子数据
        for i in mapsingle['rasa_nlu_data']['common_examples']:
            if not mapintent.has_key(i['intent']):
                mapintent.__setitem__(i['intent'], [i])
            else:
                mapintent[i['intent']].append(i)
        # 5. 子数据写入json文件
        for i in mapintent:
            mapsingleorder['rasa_nlu_data']['common_examples'].extend(mapintent[i])
            with open(os.path.join(TXTPATH, str(i)) + "-v2.json", "w") as f:
                json.dump({'rasa_nlu_data': {'common_examples': mapintent[i]}}, f, ensure_ascii=False)
        # 3. 唯一order累加写入json文件
        with open(jsonfout, "w") as f:
            json.dump(mapsingleorder, f, ensure_ascii=False)

    return setting


def drop_duplicates(incsv):
    # 1. 读文件
    # jsfile = pd.read_json(".\Shop_information.txt")
    # jsfile = pd.read_csv(incsv, encoding="utf8")
    jsfile = pd.read_csv(incsv, encoding="gb2312")
    print("total lines is:")
    print(jsfile.shape)
    # 2. 丢弃列
    # jsfile.drop("shop_id", axis=1, inplace=True)
    # 3. 某列丢弃行 重复
    # singres = jsfile["name"].drop_duplicates()
    # 3. 和并列
    # frame1.join(frame2)
    # 4. 整体丢弃行 重复
    jsfile.drop_duplicates(["text"], inplace=True)
    print("unique lines is:")
    print(jsfile.shape)
    # 5. 根据值丢弃行
    # jsfile = jsfile[(jsfile["类型"] == "其他")]
    return jsfile


def write_cont2csv(content, outcsv):
    content.to_csv(outcsv, encoding='utf-8', index=False)


def write_rasa_json(obj):
    str1 = '{"text": "'
    str2 = '","intent": "others","entities": []},'
    with open('out_json.txt', 'w', -1) as f:
        for i in obj:
            f.write(str1 + i + str2)
            f.write('\n')


def post_true_lable(filename, outname):
    """不管对不对，只取目的概率"""
    # save = pd.DataFrame({"english": "", "number": ""})
    with open(filename, 'r') as f:
        with open(outname, "w") as csvfile:
            writer = csv.writer(csvfile)
            # 先写入columns_name
            writer.writerow(["name", "text", "confidence"])
            # 写入多行用writerows
            for i1 in f:
                tmary = i1.replace(" ", "").split(";;")
                tm1 = ""
                tm2 = ""
                tm3 = ""
                tm4 = ""
                for i in tmary:
                    if i.startswith('"name":"'):
                        tm1 = i.lstrip('"name":"').rstrip('"')
                    elif i.startswith('"suggest":'):
                        tm2 = i.lstrip('"suggest":')
                    elif i.startswith('"text":"'):
                        tm3 = i.lstrip('"text":"').rstrip('",')
                    elif i.startswith('"confidence":'):
                        tm4 = float(i.lstrip('"confidence":').rstrip(',\n'))
                writer.writerows([[tm2, tm3, tm4]])


def post_sugest_lable(filename, outname):
    """参考suggest的列表"""
    # save = pd.DataFrame({"english": "", "number": ""})
    with open(filename, 'r') as f:
        with open(outname, "w") as csvfile:
            writer = csv.writer(csvfile)
            # 先写入columns_name
            writer.writerow(["name", "text", "confidence", "suggest"])
            # 写入多行用writerows
            for i1 in f:
                tmary = i1.replace(" ", "").split(";;")
                tm1 = ""
                tm2 = ""
                tm3 = ""
                tm4 = ""
                for i in tmary:
                    if i.startswith('"name":"'):
                        tm1 = i[8:].rstrip('\"')
                    elif i.startswith('"suggest":'):
                        tm2 = i[10:].rstrip('\n')
                    elif i.startswith('"text":"'):
                        tm3 = i[8:].rstrip('",\n')
                    elif i.startswith('"confidence":'):
                        tm4 = float(i[13:].rstrip(',\n'))
                writer.writerows([[tm1, tm3, tm4, tm2]])


def post_mixed(filename, outname):
    """统计所有概率"""
    jsfile = pd.read_json(filename)
    jsfile.drop(["intent", "text", "entities"], axis=1, inplace=True)
    # 1. 写转化文件
    with open(outname, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "confidence"])  # 写入多行用writerows
        for i1 in jsfile["intent_ranking"]:
            for i2 in i1:
                if i2["name"] is not None or i2["name"] != "":
                    writer.writerows([[i2["name"], i2["confidence"]]])


def plot_my(filename):
    # 2. 读转化文件
    csvf = pd.read_csv(filename)
    # csvf = csvf[(csvf["name"].notnull())]
    grouptmp = csvf[["name", "confidence"]].groupby(['name'], as_index=False)
    # grouptmp = csvf[["name", "confidence"]].groupby(['name'])
    datamean = grouptmp.mean()
    datamin = grouptmp.min()
    datamax = grouptmp.max()
    datastd = grouptmp.std()
    # 3.plot rest
    #                                   1row 2 rank     15宽  4高
    # fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    fig, (axis1, axis2, axis3) = plt.subplots(3, 1, figsize=(15, 9))
    axis1.set_title('min')
    axis2.set_title('max')
    axis3.set_title('mean')
    sns.barplot(x='name', y='confidence', data=datamin, ax=axis1)
    sns.barplot(x='name', y='confidence', data=datamax, ax=axis2)
    sns.barplot(x='name', y='confidence', data=datamean, ax=axis3)
    plt.show()


def print_config(filename):
    # 2. 读转化文件
    csvf = pd.read_csv(filename)
    grouptmp = csvf[["name", "confidence"]].groupby(['name'])
    datamean = grouptmp.mean()
    datamin = grouptmp.min()
    datamax = grouptmp.max()
    datastd = grouptmp.std()
    datacount = grouptmp.count()
    print("  count  ".center(80, "&"))
    print(datacount)
    print("  mean  ".center(80, "&"))
    print(datamean)
    print("  min  ".center(80, "&"))
    print(datamin)
    print("  max  ".center(80, "&"))
    print(datamax)
    print("  std  ".center(80, "&"))
    print(datastd)


def print_sugest_config_with_others(filesample, fileallsetcsv, fn_score=None):
    """结果标签跟目的标签一样"""
    # 1. 读转化文件
    my_matrix = np.loadtxt(open(filesample, "rt"), dtype=np.str, delimiter=",", skiprows=1)
    dimx, dimy = my_matrix.shape
    sucarr = np.arange(dimx, dtype=int)
    sucarr[:] = 0
    my_matrix = np.column_stack((my_matrix, sucarr))
    for i in range(dimx):
        # 0. 返回意图 1.请求文本 2. 返回最高概率 3. 自己的标签 4. 结果的真假
        # print(my_matrix[i, :], thresh)
        if my_matrix[i, 0] == my_matrix[i, 3]:
            my_matrix[i, 4] = 1
        elif my_matrix[i, 0] != my_matrix[i, 3]:
            my_matrix[i, 4] = 0
        else:
            raise "未考虑的数据情况"

    keyarr = np.unique(my_matrix[:, 3], return_index=False)
    mapres = {}
    for i in range(dimx):
        if not mapres.has_key(my_matrix[i, 3]):
            mapres.__setitem__(my_matrix[i, 3], 0)
            mapres.__setitem__(str(my_matrix[i, 3] + "_t"), 0)
        mapres[my_matrix[i, 3]] += 1
        if int(my_matrix[i, 4]) == 1:
            mapres[str(my_matrix[i, 3] + "_t")] += 1
    # 2. fn_score part
    if fn_score is not None:
        csvfile = pd.read_csv(fileallsetcsv, encoding="utf8")
        # csvfile = pd.read_csv(fileallsetcsv, encoding="gb2312")
        featureall = csvfile[["lable"]].count()
        all_sample_num = featureall[0]
        fn_score2 = fn_score * fn_score
        grouptmp = csvfile[["text", "lable"]].groupby(['lable'], as_index=False)
        allsetmatrix = grouptmp.count().as_matrix()
        averfnscore = 0
        correct_num = 0

    for key in keyarr:
        logger1.info(key.center(80, str("*")))
        logger1.info(key + "_sample_num: " + str(mapres[key]))
        precison = float(mapres[str(key + "_t")]) / float(mapres[key])  # 抽出的符合该标签/抽出的该标签总数
        logger1.info(key + "_precison: " + str(precison * 100) + "%")
        if fn_score is not None:
            numarry1 = [i[1] for i in allsetmatrix if i[0] == key]
            single_total_num = numarry1[0]
            recalls = mapres[key] / single_total_num  # 抽出的该标签/总测试集的该标签
            correct_num += int(mapres[str(key + "_t")])
            fn_res = (1 + fn_score2) * precison * recalls / (fn_score2 * precison + recalls)
            averfnscore += fn_res
            logger1.info(key + "-all_set_num: " + str(single_total_num))
            logger1.info(key + "-f" + str(fn_score) + "_scorn: " + str(fn_res))
    if fn_score is not None:
        averfnscore_res = averfnscore / len(keyarr)

        allprecison = float(correct_num) / float(dimx)  # 抽出的符合该标签/抽出的该标签总数
        allrecalls = dimx / all_sample_num  # 抽出的该标签/总测试集的该标签
        allkindsfnscore_res = (1 + fn_score2) * allprecison * allrecalls / (fn_score2 * allprecison + allrecalls)

        logger1.info(" total_f ".center(60, str("-")))
        logger1.info(key + "-average_f" + str(fn_score) + "_scorn: " + str(averfnscore_res))
        logger1.info(key + "-sum_kinds_f" + str(fn_score) + "_scorn: " + str(allkindsfnscore_res))


def print_sugest_config_no_others(thresh, filesample, fileallsetcsv, fn_score=None):
    """结果标签跟目的标签一样 且 目的值达到阈值"""
    # 1. 读转化文件
    my_matrix = np.loadtxt(open(filesample, "rt"), dtype=np.str, delimiter=",", skiprows=1)
    dimx, dimy = my_matrix.shape
    sucarr = np.arange(dimx, dtype=int)
    sucarr[:] = 0
    my_matrix = np.column_stack((my_matrix, sucarr))
    for i in range(dimx):
        # 0. 返回意图 1.请求文本 2. 返回最高概率 3. 自己的标签 4. 结果的真假
        # print(my_matrix[i, :], thresh)
        if ("others" == my_matrix[i, 3]) & (my_matrix[i, 0] == my_matrix[i, 3]) & (float(my_matrix[i, 2]) > thresh):
            logger1.info("model with others!!")
            my_matrix[i, 4] = 1
        elif ("others" == my_matrix[i, 3]) & (float(my_matrix[i, 2]) > thresh):
            my_matrix[i, 4] = 0
        elif ("others" == my_matrix[i, 3]) & (float(my_matrix[i, 2]) <= thresh):
            my_matrix[i, 4] = 1
        elif (my_matrix[i, 0] == my_matrix[i, 3]) & (float(my_matrix[i, 2]) > thresh):
            my_matrix[i, 4] = 1
        elif (my_matrix[i, 0] != my_matrix[i, 3]) & (float(my_matrix[i, 2]) > thresh):
            my_matrix[i, 4] = 0
        elif float(my_matrix[i, 2]) <= thresh:
            my_matrix[i, 4] = 0
        else:
            raise "未考虑的数据情况"

    keyarr = np.unique(my_matrix[:, 3], return_index=False)
    mapres = {}
    for i in range(dimx):
        if not mapres.has_key(my_matrix[i, 3]):
            mapres.__setitem__(my_matrix[i, 3], 0)
            mapres.__setitem__(str(my_matrix[i, 3] + "_t"), 0)
        mapres[my_matrix[i, 3]] += 1
        if int(my_matrix[i, 4]) == 1:
            mapres[str(my_matrix[i, 3] + "_t")] += 1
    # 2. fn_score part
    if fn_score is not None:
        csvfile = pd.read_csv(fileallsetcsv, encoding="utf8")
        # csvfile = pd.read_csv(fileallsetcsv, encoding="gb2312")
        featureall = csvfile[["lable"]].count()
        all_sample_num = featureall[0]
        fn_score2 = fn_score * fn_score
        grouptmp = csvfile[["text", "lable"]].groupby(['lable'], as_index=False)
        allsetmatrix = grouptmp.count().as_matrix()
        averfnscore = 0
        correct_num = 0

    for key in keyarr:
        logger1.info(key.center(80, str("*")))
        logger1.info(key + "_sample_num: " + str(mapres[key]))
        precison = float(mapres[str(key + "_t")]) / float(mapres[key])  # 抽出的符合该标签/抽出的该标签总数
        logger1.info(key + "_precison: " + str(precison * 100) + "%")
        if fn_score is not None:
            numarry1 = [i[1] for i in allsetmatrix if i[0] == key]
            single_total_num = numarry1[0]
            recalls = mapres[key] / single_total_num  # 抽出的该标签/总测试集的该标签
            correct_num += int(mapres[str(key + "_t")])
            fn_res = (1 + fn_score2) * precison * recalls / (fn_score2 * precison + recalls)
            averfnscore += fn_res
            logger1.info(key + "-all_set_num: " + str(single_total_num))
            logger1.info(key + "-f" + str(fn_score) + "_scorn: " + str(fn_res))
    if fn_score is not None:
        averfnscore_res = averfnscore / len(keyarr)

        allprecison = float(correct_num) / float(dimx)  # 抽出的符合该标签/抽出的该标签总数
        allrecalls = dimx / all_sample_num  # 抽出的该标签/总测试集的该标签
        allkindsfnscore_res = (1 + fn_score2) * allprecison * allrecalls / (fn_score2 * allprecison + allrecalls)

        logger1.info(" total_f ".center(60, str("-")))
        logger1.info(key + "-average_f" + str(fn_score) + "_scorn: " + str(averfnscore_res))
        logger1.info(key + "-sum_kinds_f" + str(fn_score) + "_scorn: " + str(allkindsfnscore_res))


def generate_batch_shell():
    pass


def monte_get_curl(inputname, outname, numb):
    """从总样本里提取样本，把请求内容写入文件"""
    my_matrix = np.loadtxt(open(inputname, "rt"), dtype=np.str, delimiter=",", skiprows=1)
    # res = np.random.choice(my_matrix, 5, p=[0.5, 0.1, 0.1, 0.3])
    dimx, dimy = my_matrix.shape
    indexarr = np.arange(dimx)
    indexres = np.random.choice(indexarr, size=numb, replace=False, p=None)
    randmarry = my_matrix[indexres]

    strhead = '''curl -XPOST http://10.255.232.24:5000/parse -d '{"q":"'''
    strtag = '''", "project": "shishuo", "model": "model_root_no_others"}' | jq '.' | grep -A 4 '"intent": ' | grep -E "confidence"\|"name"\|"text" | awk '{if (NR%3==0) {printf "%s ",$0} else if (NR%3==1) {printf "%s;; ",$0} else {printf "%s;; \\"suggest\\":'''
    strtail = ''';;",$0}}\' ; echo ""'''
    with open(outname, "wt") as f:
        for i1 in range(numb):
            temstr = strhead.encode('utf8') + randmarry[i1, 0] + strtag.encode('utf8') + randmarry[
                i1, 1] + strtail.encode('utf8')
            f.write(temstr)
            f.write('\n')


def monte_get_direct(inputname, outname, numb):
    """从总样本里提取样本，请求后吧结果写入文件"""
    my_matrix = np.loadtxt(open(inputname, "rt"), dtype=np.str, delimiter=",", skiprows=1)
    # res = np.random.choice(my_matrix, 5, p=[0.5, 0.1, 0.1, 0.3])
    dimx, dimy = my_matrix.shape
    indexarr = np.arange(dimx)
    indexres = np.random.choice(indexarr, size=numb, replace=False, p=None)
    randmarry = my_matrix[indexres]
    get_data_from_array(randmarry, outname)


def test_no_entity(inputname, outname):
    """从总样本里提取样本，请求后吧结果写入文件"""
    test_array = []
    with open(inputname, "rt") as csvfile:
        res = csv.reader(csvfile)
        for i in res:
            try:
                test_array.append(i)
            except Exception as e:
                print()
    with open(outname, "wb") as csvfile:
        writer = csv.writer(csvfile)
        url = "http://10.255.232.24:5000"
        api = "/parse"
        for stri in test_array:
            # # a = ShishuoApi().get_rasa_intents(stri.decode('gbk'), "shishuo", "model_stock")
            # # if len(a["entities"]) == 0:
            # #     writer.writerow([stri])
            # a = ShishuoApi().get_rasa_intents(url, api, stri.decode('gbk'), "shishuo", "model_root")
            # if a["intent"]["name"] != "stock":
            #     writer.writerow([stri])
            a = ShishuoApi().get_rasa_intents(url, api, stri[1].decode('gbk'), "shishuo", "model_root")
            if a["intent"]["name"] != stri[0]:
                writer.writerow([a["intent"]["name"], a["text"]])


def get_data_from_array(data_target_array, outname):
    """请求数据的结果写到文件里"""
    with open(outname, "wt") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "text", "confidence", "suggest"])
        for stri in data_target_array:
            # a = ShishuoApi().get_rasa_intents(stri[0], "shishuo", "model_root_no_others")
            a = ShishuoApi().get_rasa_intents(stri[0], "shishuo", "model_root")
            writer.writerows([[a["intent"]["name"], a["text"], a["intent"]["confidence"], stri[1]]])


def wrong_result(inputname, outname):
    """错误打印"""
    my_matrix = np.loadtxt(open(inputname, "r"), dtype=np.str, delimiter=",", skiprows=1)
    with open(outname, "w") as csvout:
        writer = csv.writer(csvout)
        writer.writerow(["name", "text", "confidence", "suggest"])
        for stri in my_matrix:
            if stri[0] != stri[3]:
                writer.writerows([[stri[0], stri[1], stri[2], stri[3]]])


def look4keys(res):
    # with codecs.open(filename, 'r', encoding="UTF-8") as f:
    #     res = json.load(f)
    #     res = res['rasa_nlu_data']['common_examples']
    intentset = set()
    entityset = set()
    for item in res:
        if item['intent'] not in intentset:
            intentset.add(item['intent'])
        for entity in item['entities']:
            if entity['entity'] not in entityset:
                entityset.add(entity['entity'])
    print('intent:', intentset)
    print('entity:', entityset)


if __name__ == '__main__':
    bpath = os.path.join("g:\\", "project", "Rasa_NLU_Chi", "data", "sz_base")
    files = os.listdir(bpath)
    aljson = []
    for file1 in files:
        with codecs.open(os.path.join(bpath, file1), 'r', encoding="UTF-8") as f:
            res = json.load(f)
            res = res['rasa_nlu_data']['common_examples']
            # if 'rasa_nlu_data' not in aljson:
            #     aljson['rasa_nlu_data'] = {}
            # if 'common_examples' not in aljson['rasa_nlu_data']:
            #     aljson['rasa_nlu_data']['common_examples'] = []
            aljson += res
    look4keys(aljson)
    exit()
    rootjson = os.path.join(TXTPATH, "rasa_zh_cookbook.json")
    logjson = os.path.join(TXTPATH, "rasa_zh_root22.json")
    csvin = os.path.join(TXTPATH, "logout_tmp.csv")
    csvout = os.path.join(TXTPATH, "logout_test.csv")
    # csvoutbak = os.path.join(TXTPATH, "logout_bak.csv")
    csvoutbak = os.path.join(TXTPATH, "sample_res5.csv")
    # txtfin = os.path.join(TXTPATH, "stock_name.txt")
    txtfin = os.path.join(TXTPATH, "test_in.csv")
    # jsonfin = os.path.join(TXTPATH, "tmp.json")
    jsonfin = os.path.join(TXTPATH, "rasa_zh_root.json")
    jsonfout = os.path.join(TXTPATH, "rasa_zh_root_single.json")
    all_sample_csv = os.path.join(TXTPATH, "simple_set.csv")
    single_sample_csv = os.path.join(TXTPATH, "single_set.csv")
    sample_res_csv = os.path.join(TXTPATH, "sample_res.csv")
    error_csv = os.path.join(TXTPATH, "error_res.csv")
    monte_num = 500

    # # 0. 训练json文件去重, 按intent分离
    # txt2json(txtfin, jsonfin)
    # # 0. 训练json文件去重, 按intent分离
    # drop_duplicate_json(jsonfin, jsonfout)

    # # 去重log包含的json
    # log2csv(logjson, csvin)
    # # 训练json2csv intent
    # # json2csv(rootjson, csvout)
    # json2csv(logjson, csvin)
    # # 手动去重整csv
    # csvddupli(csvin, csvout)
    # 识别csv错误标签后 生产 主子新json，手动加入数据
    # csv2json(csvout)
    csv21json(csvout)
    # 手动去重整合到训练json的数据。
    # log2csv(logjson, csvout)

    # # 1. 测试数据简化
    # content = drop_duplicates(all_sample_csv)
    # write_cont2csv(content, single_sample_csv)
    #
    # # 2. 随机shell
    # # get_data_from_array([["今天熊市还是牛市", "stock"]], "test.csv")
    # # monte_get_curl("single_set.csv","monte.sh", 100)
    # # monte_get_direct(single_sample_csv, sample_res_csv, monte_num)
    # test_no_entity(txtfin, sample_res_csv)
    # #
    # # # 3. 真值转结果
    # # # post_true_lable('monte.txt', 'monte_lable.csv')
    # # # 4. 混态转结果
    # # # post_mixed('mixed.txt','mixed.csv')
    # #
    # # # # 5. 有参考sugget的输出
    # # # post_sugest_lable('monte.txt', 'monte_lable.csv')
    # #
    # # # 6. 有参考sugget的输出
    # # print_sugest_config_with_others(sample_res_csv, single_sample_csv, fn_score=1)
    #
    # # logger1.info(" thresh 0.001  ".center(100, str("*")))
    # # print_sugest_config_no_others(0.001, sample_res_csv, single_sample_csv, fn_score=1)
    # # logger1.info("")
    # # logger1.info("")
    # # logger1.info(" thresh 0.3  ".center(100, str("*")))
    # # print_sugest_config_no_others(0.3, sample_res_csv, single_sample_csv, fn_score=1)
    # # logger1.info("")
    # # logger1.info("")
    # # logger1.info(" thresh 0.4  ".center(100, str("*")))
    # # print_sugest_config_no_others(0.4, sample_res_csv, single_sample_csv, fn_score=1)
    # # logger1.info("")
    # # logger1.info("")
    # # logger1.info(" thresh 0.5  ".center(100, str("*")))
    # # print_sugest_config_no_others(0.5, sample_res_csv, single_sample_csv, fn_score=1)
    # # logger1.info("")
    # # logger1.info("")
    # # logger1.info(" thresh 0.6  ".center(100, str("*")))
    # # print_sugest_config_no_others(0.6, sample_res_csv, single_sample_csv, fn_score=1)
    # #
    # # # 7.绘图
    # # print("mixed".center(100,"*"))
    # # print_config('mixed.csv')
    # # print("ture_lable".center(100,"*"))
    # # print_config('post_lable.csv')
    # # # write_rasa_json(res["内容"])
    #
    # # 8. 识别错误的输出
    # # wrong_result(sample_res_csv, error_csv)
