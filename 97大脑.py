import pandas as pd
import numpy as np
import re
import json
import os
import jieba

pathf = os.path.join("..", "data")
pathv = os.path.join("..", "..", "nlp", "nocode", "sys", "wordvector", "wiki.zh.vec")
stopwords_path = os.path.join("..", "..", "nlp", "ai_sets", "dict", "_stop_dict_.txt")
stop_set = set([value.replace('\n', '') for value in open(stopwords_path, 'r', encoding='utf8').readlines()])
stop_set.add("目前")
stop_set.add("公司")
stop_set.add("主要")
stop_set.add("从事")
stop_set.add("需要")
stop_set.add("和")
stop_set.add("基于")
stop_set.add("所")
stop_set.add("即")
stop_set.add("有")
stop_set.add("是")
stop_set.add("总")
stop_set.add("及")
stop_set.add("包括")
stop_set.add("还有")
stop_set.add("的")
stop_set.add("无")
stop_set.add("等")
stop_set.add("等等")
stop_set.add("服务")
stop_set.add("行业")
stop_set.add("现在")
stop_set.add("受限于")
stop_set.add("以及")
stop_set.add("过程")
stop_set.add("产品")
stop_set.add("能")
stop_set.add("多")
stop_set.add("想")
stop_set.add("扩大")
stop_set.add("多")
stop_set.add("在")
stop_set.add("项目")
stop_set.add("没有")
stop_set.add("1")
stop_set.add("2")
stop_set.add("3")
stop_set.add("+")
stop_set.add("-")
stop_set.add("用")


def getdata():
    analynpd = pd.read_excel(io=os.path.join(pathf, "面试数据.xlsx"), sheet_name='Sheet1', header=1)
    print(analynpd.columns)
    print(analynpd.head(2))
    print(analynpd.shape)
    return analynpd


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


def fenci_clean(strings):
    a = jieba.cut(strings)
    a = [i1.replace("\n", "") for i1 in a]
    a = [i1.replace(",", "") for i1 in a]
    a = [i1.replace("，", "") for i1 in a]
    a = [i1.replace("。", "") for i1 in a]
    a = [i1.replace(".", "") for i1 in a]
    a = [i1.replace(";", "") for i1 in a]
    a = [i1.replace("；", "") for i1 in a]
    a = [i1.replace("、", "") for i1 in a]
    a = [i1.replace("【", "") for i1 in a]
    a = [i1.replace("】", "") for i1 in a]
    a = [i1.replace("[", "") for i1 in a]
    a = [i1.replace("]", "") for i1 in a]
    a = [i1.replace("(", "") for i1 in a]
    a = [i1.replace(")", "") for i1 in a]
    a = [i1.replace("（", "") for i1 in a]
    a = [i1.replace("）", "") for i1 in a]
    a = [i1.replace(" ", "") for i1 in a]
    a = [i1.strip() for i1 in a]
    return [i1 for i1 in a if i1 != "" and i1 is not None and i1 not in stop_set]


def find_common(desire, product):
    retB = list(set(desire).intersection(set(product)))
    if len(set(desire))>0:
        numb = len(retB) / len(set(desire))
    else:
        numb = 0
    if numb > 0.1:
        return 1, numb
    else:
        return 0, 0


def analyfunc(analynpd):
    # 1. 分类聚合
    # 2.
    analynpd["可销售的产品和服务"] = analynpd["可销售的产品和服务"].map(fenci_clean)
    analynpd["需要采购的产品和服务"] = analynpd["需要采购的产品和服务"].map(fenci_clean)
    reslist = []
    for i1 in analynpd.index:
        tmpperon = {}
        for i2 in analynpd.index:
            sig, numb = find_common(analynpd.loc[i1, "需要采购的产品和服务"], analynpd.loc[i2, "可销售的产品和服务"])
            if 1 == sig:
                tmpperon[analynpd.loc[i2, "联系人"] + "_" + str(i2)] = numb
        tmpperonl = sorted(tmpperon.items(), key=lambda x: -x[1])
        tmpperons = [i1[0] + "_" + str(int(i1[1] * 100)) + "%" for i1 in tmpperonl]
        reslist.append([analynpd.loc[i1, "联系人"] + "_" + str(i1), tmpperons])
    respd = pd.DataFrame(reslist)
    respd.columns = ["需求人", "供应人列表"]
    respd.to_excel(os.path.join("..", "联系人列表.xlsx"), sheet_name='Sheet1', index=False)


def main():
    analynpd = getdata()
    analyfunc(analynpd)


if __name__ == "__main__":
    main()
