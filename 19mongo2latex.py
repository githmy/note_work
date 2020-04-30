import pandas as pd
import numpy as np
import re
import sys
import itertools
import string
from sklearn.utils import shuffle


def iterjson(source):
    reslist = []
    if type(source) is dict:
        for v in source.values():
            reslist += iterjson(v)
    elif type(source) is list:
        for v in source:
            reslist += iterjson(v)
    elif type(source) is str:
        reslist = [item.strip("$") for item in re.findall(r"\$.*?\$", source, re.U | re.M)]
        reslist = [i1 for i1 in reslist if "<%=" not in i1]
        # if "=\\left(\\sin ^{2}" in source:
        #     sys.stdin.readline()
        eesig = 0
        # ①②③④⑤⑥⑦⑧⑨⑩
        for i1 in reslist:
            # if "β" in i1:
            #     print("%%%%%%%%%%%%%%%")
            #     print(source)
            #     print(reslist)
            #     sys.stdin.readline()
            for ch in i1:
                if u'\u4e00' <= ch <= u'\u9fff' or ch in "①②③④⑤⑥⑦⑧⑨⑩αβγ∠：，“”△▪•":
                    eesig = 1
                    break
            if eesig == 1:
                reslist = []
                break
    else:
        pass
    return reslist


def main():
    from utils.connect_mongo import MongoDB
    quire_list = ["examples", "questions"]
    config_new = {
        'host': "127.0.0.1",
        'port': 27017,
        'database': "thinking2-test",
        'col': "examples",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
    }
    ins_new = MongoDB(config_new)
    res_map_new = ins_new.exec_require(quire_list)
    contente = iterjson(list(res_map_new["examples"]))
    contentq = iterjson(list(res_map_new["questions"]))
    contenta = contente + contentq
    contenta = [i1.strip() for i1 in contenta]
    dictt = {i1: len(i1) for i1 in set(contenta)}
    contenta = sorted(dictt.items(), key=lambda x: x[1])
    contenta = [i1[0] for i1 in contenta]
    # contenta = list(set(contenta))
    # contenta.sort()
    data = pd.DataFrame(contenta)
    data.to_excel('../formula.xls', sheet_name='Sheet1', index=False)
    print(234)
    print(len(contenta))
    print(222)
    print(contenta[0:20])
    print(contenta[-20:])
    pass


def latex2space():
    data = pd.read_excel(io='../formula.xls', sheet_name='Sheet1', header=None)
    tk = pd.read_csv('../token.csv', header=None)
    listtk = {}
    for i1 in tk[0]:
        tt = i1.replace(" ", "")
        tlsit = []
        for i2 in tt:
            tlsit.append(i2)
        listtk[" ".join(tlsit)] = len(" ".join(tlsit))
    contenta = sorted(listtk.items(), key=lambda x: -x[1])
    contenta = [i1[0] for i1 in contenta]
    contento = [i1.replace(" ", "") for i1 in contenta]

    def splitt(strings):
        lists = []
        for i1 in str(strings):
            lists.append(i1)
        tstr = " ".join(lists)
        for s, n in zip(contenta, contento):
            tstr = tstr.replace(s, n)
        tstr = tstr.replace("  ", " ").replace("  ", " ")
        return tstr

    data[1] = data[0].map(splitt)
    tkens = set(itertools.chain(*[i1.split(" ") for i1 in data[1]]))
    for i1 in string.ascii_letters + string.digits + string.punctuation + string.ascii_letters:
        tkens.add(i1)
    for i1 in tk[0]:
        tkens.add(i1)
    print(tkens)
    print(len(tkens))
    newtk = pd.DataFrame(tkens)
    newtk.to_csv('../token_new.csv', index=False, header=None, encoding="utf-8")
    data.to_excel('../formula_new.xls', sheet_name='Sheet1', index=False, header=None)


def filttk():
    # with open('../token.txt', 'rt', encoding="utf8") as f:
    with open('../token.txt', 'rt') as f:
        result = f.readlines()
        result = [i1.strip("\n") for i1 in result]
        print(result)
    data = pd.read_excel(io='../formula_new.xls', sheet_name='Sheet1', header=None)
    for i1 in data[1]:
        for i2 in i1.split(" "):
            if i2 not in result:
                print(i1 + "#" + i2 + "#")
                # tkens = set(itertools.chain(*[i1.split(" ") for i1 in data[1]]))
                # for i1 in tkens:
                #     if i1 not in result:
                #         print(i1)


def genfinal():
    data = pd.read_excel(io='../formula_new.xls', sheet_name='Sheet1', header=None)
    print(data[1])
    data = shuffle(data).reset_index(drop=True)
    print(data[1])
    filename = '../formula.txt'
    with open(filename, 'wt', encoding="utf8") as f2:
        for i1 in data[1]:
            f2.write(i1 + "\n")
    data.to_csv('../formula_hand.csv', index=True, header=None, encoding="utf-8")


if __name__ == "__main__":
    # main()
    # latex2space()
    # filttk()
    # genfinal()
    pass
