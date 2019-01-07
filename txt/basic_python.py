import jieba
import pandas as pd

# 警告过滤
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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


if __name__ == '__main__':
    # cut_files()
    n = 3000
    static_top_n(n)
