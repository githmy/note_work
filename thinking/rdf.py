# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
from server import Delphis
from models.model_cnn import TextCNN
from utils.data_trans import data2js
from utils.log_tool import logger
from sklearn.cluster import KMeans
import os
import jieba
from utils.connect_mysql import MysqlDB
import pymysql

pd.options.display.max_columns = 999

"""
1. 知识抽取
1.1 实体抽取
1.2 语义类抽取
1.2.1 并列度相似计算 两个词的相似性
1.2.2 上下位关系提取
1.2.3 语义类生成 
1.3 属性和属性值抽取
1.4 关系抽取
2. 知识表示
2.1 代表模型
2.2 复杂关系模型
3. 知识融合
3.1 实体对齐
3.2 知识加工
4. 知识推理
"""

class RDF(object):
    def __init__(self):
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "thinkingdata",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.mysql = MysqlDB(config)
        self.update_triples_from_db()

    # 1. 根据 实体 找到相关 的属性
    def find_entity_property(self, entityname):
        sqls = """
          select SubjectName from char_words_triple WHERE wordrelationname='属性' AND wordrelationid=3 AND objectname in 
          (select objectname from char_words_triple WHERE wordrelationname='属性' AND wordrelationid=3 AND SubjectName='{}');
        """.format(entityname)
        sqlres = self.mysql.exec_sql(sqls)
        return sqlres

    # 2. 根据 三元组 主体 找到相关 的属性
    def find_triple_subject_property(self, entityname, tripleid):
        pass

    # 3. 根据 三元组 客体 找到相关 的属性
    def find_triple_object_property(self, entityname, tripleid):
        pass

    # 4. 根据 三元组 场景 找到相关 的属性
    def find_triple_scene_property(self, entityname, tripleid):
        pass

    # 5. 获取相关 属性的 实体
    def find_relate_property_of_entities(self, tripleid):
        pass

    # 6. 根据 要求 条件演化
    def inference_base_triple(self, tripleid):
        pass

    # 2. 句子导入三元组
    def sentences2triples(self, sentence_list):
        res_list = []
        for i1 in sentence_list:
            words = jieba.cut(i1["sentence"])
            res_list.append(list(words))
        return res_list

    # 3. 更新 三元组
    def update_triples_from_db(self):
        self.triple_obj = [
            {"subject": "", "relation": "", "object": "", "weigh": ""},
            {"subject": "", "relation": "", "object": "", "weigh": ""},
        ]


def main():
    ins = RDF()
    entityname = "单项式"
    find_list = ins.find_entity_property(entityname)
    print(find_list)
    print(pd.DataFrame(find_list))
    test_obj = [
        {"sentence": "一个月内，小明体重增加 $2kg$，小华体重减少 $1kg$，小强体重无变化，写出他们这个月的体重增长值"},
        {
            "sentence": "$2001$ 年下列国家的商品进出口总额比上年的变化情况是：\ 美国减少 $6.4\%$，德国增长 $1.3\%$，\ 法国减少 $2.4\%$，英国减少 $3.5\%$，\ 意大利增长 $0.2\%$，中国增长 $7.5\%$ ，\ 写出这些国家 $2001$ 年商品进出口总额的增长率"},
        {"sentence": "高出海平面记为正 ，低于海平面记为负 ，若地图上 $A$ ，$B$ 两地的高度分别标记为 $4600$ 米和 $-200$ 米 ，你能说出它们的含义吗"},
    ]
    discrete_list = ins.sentences2triples(test_obj)
    print(discrete_list)


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to Delphis".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    main()
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
