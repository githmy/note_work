# coding: utf-8
import jieba, re
import jieba.posseg as pseg
import sys, unicodedata


def text_cut(s_path, d_path, u_path, dataset):
    """
    对清洗后的文本进行分词
    :param s_path:停用词典路径；d_path:删除词典路径；u_path:用户词典路径；dataset: 要分词的数据，数据框格式
    :return: 分词后的数据，数据框格式（dataset会增加'word_cut'列）
    """
    # 停用词
    stopwords_path = s_path
    stop_set = set([value.replace('\n', '') for value in open(stopwords_path, 'r', encoding='utf8').readlines()])
    # jieba分词包删除词
    del_path = d_path
    del_ls = [value.replace('\n', '') for value in open(del_path, encoding='utf8').readlines()]
    for i in del_ls:
        jieba.del_word(i)

    # 加载用户词典
    userdict_path = u_path
    jieba.load_userdict(userdict_path)

    # 保留词性
    flag_ls = ['a', 'ad', 'b', 'd', 'f', 'i', 'l', 'm', 'n', 'nrt', 'ns', 'nt', 'nz', 'v', 'vn', 'x']

    # 分词并筛词
    def pseg_cut(text):
        words = pseg.cut(text)
        return ' '.join([w.word for w in words if w.flag in flag_ls and w.word not in stop_set and len(w.word) >= 2])

    word_list = [i for i in dataset.columns if i.startswith("word_") and i.endswith("__")]
    for word in word_list:
        dataset[word + "cut_"] = dataset[word].map(pseg_cut)

    # 清洗用的正则表达式
    res = re.compile(r'\s+')
    red = re.compile(r'^(\d+)$')

    # 清洗标点符号等异常字符
    todel = dict.fromkeys(i for i in range(sys.maxunicode)
                          if unicodedata.category(chr(i)) not in ('Lu', 'Ll', 'Lt', 'Lo', 'Nd', 'Nl', 'Zs'))

    # 清洗分词结果的方法
    def cleantext(text):
        # try:
        #     text = unicode(text)
        # except:
        #     pass
        if text != '':
            return re.sub(res, ' ',
                          ' '.join(map(lambda x: re.sub(red, '', x), text.translate(todel).split(' ')))).strip()
        else:
            return text

    # 对分词结果进行清洗
    for word in word_list:
        dataset[word + "cut_"] = dataset[word].map(pseg_cut)
        dataset[word + "cut_"] = dataset[word + "cut_"].map(cleantext)

    return dataset
