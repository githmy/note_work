# coding: utf-8
from utils.t_fanjian import Traditional2Simplified


# 遍历，不要固定字段
def text_clean(r_path, dataset):
    """
    清洗现有数据
    :param path: 替换词典路径；
    :param path: dataset: 要清洗的数据，数据框格式
    :return: 清洗后的数据，数据框格式（dataset增加列'word_text'）
    """
    word_list = [i for i in dataset.columns if i.startswith("word_")]
    for word in word_list:
        dataset = dataset[dataset[word].notnull()]
        dataset.index = range(len(dataset))

        # 把描述中的英文字母全部变成小写的
        dataset[word + "__"] = dataset[word].str.lower()

        # 把文本中的繁体字转化为简体字
        dataset[word + "__"] = dataset[word + "__"].map(Traditional2Simplified)

        # 错别字、同义词/近义词替换
        words_replace_path = r_path
        f = open(words_replace_path, 'r', encoding='utf8')
        for line in f.readlines():
            value = line.strip().replace('\n', '').split(',')
            dataset[word + "__"] = dataset[word + "__"].str.replace(value[0], value[1])
    return dataset
