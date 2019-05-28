import jieba
from jieba import posseg

# 警告过滤
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def fenci(filename):
    stringrr = "普通分词和词性标注分词"
    res1 = jieba.cut(stringrr)
    print(list(res1))
    res2 = posseg.cut(stringrr)
    print(list(res2))



if __name__ == '__main__':
    # cut_files()
    n = 3000
    fenci(n)
