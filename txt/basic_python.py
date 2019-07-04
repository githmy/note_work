import jieba
import pandas as pd

# 警告过滤
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# !pip install goto-statement
from goto import with_goto


@with_goto
def goto_demo():
    i = 1
    result = []
    label.begin
    if i == 2:
        goto.end

    result.append(i)
    i += 1
    goto.begin
    label.end


def list_deal():
    listA = [1, 2, 3, 4, 5]
    listB = [3, 4, 5, 6, 7]
    # 逆序列
    listB.reverse()
    # # 元素的序号
    # your_list.index('your_item')
    # your_list.sort(cmp=None, key=None, reverse=False)
    # 交集
    retA = [i for i in listA if i in listB]
    retB = list(set(listA).intersection(set(listB)))
    print("retA is: ", retA)
    print("retB is: ", retB)
    # 求并集
    retC = list(set(listA).union(set(listB)))
    print("retC1 is: ", retC)

    # 求差集，在B中但不在A中
    retD = list(set(listB).difference(set(listA)))
    print("retD is: ", retD)
    retE = [i for i in listB if i not in listA]
    print("retE is: ", retE)

    dict_data = {6: 109, 10: 105, 3: 211, 8: 102, 7: 106}


def json_sort():
    # json 字典排序
    dict_data = {6: 109, 10: 105, 3: 211, 8: 102, 7: 106}

    # 对字典按键（key）进行排序（默认由小到大）
    test_data_0 = sorted(dict_data.keys())
    # 输出结果
    print(test_data_0)  # [3, 6, 7, 8, 10]
    test_data_1 = sorted(dict_data.items(), key=lambda x: x[0])
    # 输出结果
    print(test_data_1)  # [(3, 211), (6, 109), (7, 106), (8, 102), (10, 105)]

    # 对字典按值（value）进行排序（默认由小到大）
    test_data_2 = sorted(dict_data.items(), key=lambda x: x[1])
    # 输出结果
    print(test_data_2)  # [(8, 102), (10, 105), (7, 106), (6, 109), (3, 211)]
    test_data_3 = sorted(dict_data.items(), key=lambda x: x[1], reverse=True)
    # 输出结果
    print(test_data_3)  # [(3, 211), (6, 109), (7, 106), (10, 105), (8, 102)]


def cut_one_file(filename):
    print("cuting:" + filename)
    with open("bcut." + filename + ".bcut", 'rt', encoding="utf8") as f:
        result = f.readlines()
        with open(filename, 'wt', encoding="utf8") as f2:
            for i in result:
                f2.write(" ".join(jieba.cut(i)))


def cut_files():
    cut_one_file("dev.zh")
    cut_one_file("train.zh")
    cut_one_file("tst.zh")


def static_top_n(n):
    arrayobjs = {}

    def countrer(filename):
        with open(filename, 'rt', encoding="utf8") as f:
            results = f.readlines()
            for result in results:
                arry1 = result.split(" ")
                for ele1 in arry1:
                    if ele1 not in arrayobjs:
                        arrayobjs.__setitem__(ele1, 0)
                    arrayobjs[ele1] += 1

    # 生成排序
    filename = "dev.zh"
    countrer(filename)
    filename = "train.zh"
    countrer(filename)
    filename = "tst.zh"
    countrer(filename)

    # 不在字典的排除
    with open("bak.vocab.zh.bak", 'rt', encoding="utf8") as f:
        resultv = f.readlines()
    resultv = [i.rstrip("\n") for i in resultv]

    print(len(arrayobjs))
    arraynews = {i[0]: i[1] for i in arrayobjs.items() if i[0] in resultv}
    print(len(arraynews))

    # 排序后最大的
    resdic = sorted(arrayobjs.items(), key=lambda x: x[1])
    with open("vocab.zh", 'wt', encoding="utf8") as f2:
        for i in resdic.items():
            f2.write(i[0])


def regex():
    import re
    dir_name = "abv ADF：我们"
    # 1. 直接替换
    # 非英文和数字之后的替换为空
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    print(label_name)
    # 2. 先编译再替换
    p = re.compile(r'[^a-z0-9]+')
    # 非英文和数字之后的替换为空,最多2次
    label_name = p.sub(' ', dir_name.lower(), 2)
    print(label_name)
    # 3. 正则匹配 输出前n个
    pattern = re.compile(r'\d+')  # 查找数字
    result2 = pattern.findall('run88oob123google456', 0, 6)
    print(result2)


def itertools():
    # 1. 排列组合
    shelf_list = [1, 2, 3]
    combnum = 4
    for i2 in itertools.combinations(shelf_list, combnum):
        print(i2)
    # 2. lists 合并
    batchids = list(itertools.chain(*[[i1, i1, i1, i1] for i1 in range(9)]))
    # 3. json 合并
    batchbetch_all = {"19": 2, "31": 6}
    batchbetch1 = {"1": 2, "9": 2, "3": 4, "5": 6}
    batchbetch_all.update(batchbetch1)


def json_manipulate():
    import json
    tmpjson = {"a": 1, "b": 2}
    # 不用ascii编码，缩进4格
    jsonstr = json.dumps(tmpjson, ensure_ascii=False, indent=4)
    # 用utf-8编码，缩进4格
    jsonobj = json.loads(jsonstr, encoding="utf-8")



if __name__ == '__main__':
    # cut_files()
    json_sort()
    n = 3000
    static_top_n(n)
