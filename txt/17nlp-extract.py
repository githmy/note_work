# -*- encoding:utf-8 -*-
from __future__ import print_function

import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence


def key_word_sentence():
    text = codecs.open('../test/doc/01.txt', 'r', 'utf-8').read()
    tr4w = TextRank4Keyword()

    tr4w.analyze(text=text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    print('关键词：')
    for item in tr4w.get_keywords(20, word_min_len=1):
        print(item.word, item.weight)

    print()
    print('关键短语：')
    for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=2):
        print(phrase)

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')

    print()
    print('摘要：')
    for item in tr4s.get_key_sentences(num=3):
        print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重


def kind4_data_sets():
    # -*- encoding:utf-8 -*-
    from __future__ import print_function
    import codecs
    from textrank4zh import TextRank4Keyword, TextRank4Sentence

    import sys
    try:
        reload(sys)
        sys.setdefaultencoding('utf-8')
    except:
        pass

    text = "这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足。答谢宴于晚上8点开始。"
    tr4w = TextRank4Keyword()

    tr4w.analyze(text=text, lower=True, window=2)

    print()
    print('sentences:')
    for s in tr4w.sentences:
        print(s)  # py2中是unicode类型。py3中是str类型。

    print()
    print('words_no_filter')
    for words in tr4w.words_no_filter:
        print('/'.join(words))  # py2中是unicode类型。py3中是str类型。

    print()
    print('words_no_stop_words')
    for words in tr4w.words_no_stop_words:
        print('/'.join(words))  # py2中是unicode类型。py3中是str类型。

    print()
    print('words_all_filters')
    for words in tr4w.words_all_filters:
        print('/'.join(words))  # py2中是unicode类型。py3中是str类型。

def MMR():
    # -*- coding: utf-8 -*-
    """
    Created on Thu Sep  7 17:10:57 2017
    @author: Mee
    """

    import os
    import re
    import jieba
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import operator

    f = open(r'C:\Users\user\Documents\Python Scripts/stopword.dic')  # 停止词
    stopwords = f.readlines()
    stopwords = [i.replace("\n", "") for i in stopwords]

    def cleanData(name):
        setlast = jieba.cut(name, cut_all=False)
        seg_list = [i.lower() for i in setlast if i not in stopwords]
        return " ".join(seg_list)

    def calculateSimilarity(sentence, doc):  # 根据句子和句子，句子和文档的余弦相似度
        if doc == []:
            return 0
        vocab = {}
        for word in sentence.split():
            vocab[word] = 0  # 生成所在句子的单词字典，值为0

        docInOneSentence = '';
        for t in doc:
            docInOneSentence += (t + ' ')  # 所有剩余句子合并
            for word in t.split():
                vocab[word] = 0  # 所有剩余句子的单词字典，值为0

        cv = CountVectorizer(vocabulary=vocab.keys())

        docVector = cv.fit_transform([docInOneSentence])
        sentenceVector = cv.fit_transform([sentence])
        return cosine_similarity(docVector, sentenceVector)[0][0]

    data = open(r"C:\Users\user\Documents\Python Scripts\test.txt")  # 测试文件
    texts = data.readlines()  # 读行
    texts = [i[:-1] if i[-1] == '\n' else i for i in texts]

    sentences = []
    clean = []
    originalSentenceOf = {}

    import time
    start = time.time()

    # Data cleansing
    for line in texts:
        parts = line.split('。')[:-1]  # 句子拆分
        #	print (parts)
        for part in parts:
            cl = cleanData(part)  # 句子切分以及去掉停止词
            #		print (cl)
            sentences.append(part)  # 原本的句子
            clean.append(cl)  # 干净有重复的句子
            originalSentenceOf[cl] = part  # 字典格式
    setClean = set(clean)  # 干净无重复的句子

    # calculate Similarity score each sentence with whole documents
    scores = {}
    for data in clean:
        temp_doc = setClean - set([data])  # 在除了当前句子的剩余所有句子
        score = calculateSimilarity(data, list(temp_doc))  # 计算当前句子与剩余所有句子的相似度
        scores[data] = score  # 得到相似度的列表
    # print score


    # calculate MMR
    n = 25 * len(sentences) / 100  # 摘要的比例大小
    alpha = 0.7
    summarySet = []
    while n > 0:
        mmr = {}
        # kurangkan dengan set summary
        for sentence in scores.keys():
            if not sentence in summarySet:
                mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence,
                                                                                             summarySet)  # 公式
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summarySet.append(selected)
        #	print (summarySet)
        n -= 1

    # rint str(time.time() - start)

    print('\nSummary:\n')
    for sentence in summarySet:
        print(originalSentenceOf[sentence].lstrip(' '))
    print('=============================================================')
    print('\nOriginal Passages:\n')


if __name__ == "__main__":
    key_word_sentence()
    kind4_data_sets()
    MMR()
