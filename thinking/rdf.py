# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
import os
import jieba
from utils.log_tool import *
from utils.connect_mysql import MysqlDB
import pymysql
import neo4j
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

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

config = {
    'host': "127.0.0.1",
    'user': "root",
    'password': "333",
    'port': 3306,
    'database': "thinkingdata",
    'charset': 'utf8mb4',  # 支持1-4个字节字符
    'cursorclass': pymysql.cursors.DictCursor
}


class RDF(object):
    def __init__(self, config, data=None):
        self.mysql = MysqlDB(config)
        self.update_triples_from_db()
        self.data = data

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
    ins = RDF(config)
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


def test_neo4j():
    # 图库驱动
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))

    def add_friend(tx, name, friend_name):
        tx.run("MERGE (a:Person {name: $name}) "
               "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
               name=name, friend_name=friend_name)

    def print_friends(tx, name):
        for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                             "RETURN friend.name ORDER BY friend.name", name=name):
            print(record["friend.name"])

    def add_obj(tx, name):
        tx.run("MERGE (a:obj {name: $name}) ",
               name=name)

    def add_relation(tx, subject, relation, object, domain=None):
        tx.run("MERGE (a:obj {name: $subject_name}) "
               "MERGE (a)-[r:%s {domain:$domain_name}]->(b:obj {name: $object_name})" % (relation),
               subject_name=subject, object_name=object, domain_name=domain)
        # tx.run("MERGE (a:obj {name: $subject_name}) "
        #        "MERGE (a)-[r:%s]->(b:obj {name: $object_name})" % relation,
        #        subject_name=subject, object_name=object)

    def delete_all(tx):
        "start n=node(*)  match (n)-[r:observer]-()  delete n,r match (o:org{name: 'juxinli'}) match (n)-[r:observer]-()  delete o,r"
        tx.run("match (n) detach delete n")

    with driver.session() as session:
        # session.write_transaction(add_friend, "Arthur", "Guinevere")
        # session.write_transaction(add_friend, "Arthur", "Lancelot")
        # session.write_transaction(add_friend, "Arthur", "Merlin")
        # session.read_transaction(print_friends, "Arthur")
        # session.write_transaction(add_obj, "ab")
        # 1. 清空原有
        session.write_transaction(delete_all)
        # 2. 获取数据
        filename = os.path.join(project_path, "graph.xlsx")
        pdobj = pd.read_excel(filename, sheet_name='Sheet1', header=0)
        for i1 in pdobj.iterrows():
            # session.write_transaction(add_relation, "个", "键", "安")
            session.write_transaction(add_relation, i1[1][0], i1[1][2], i1[1][4], "教育")
    # print(pdobj)
    exit()
    ins = RDF(config, pdobj)
    entityname = "单项式"
    find_list = ins.find_entity_property(entityname)
    # pdobj = pd.DataFrame(os.path.join(data_path, "graph.xlsx"))


def test_networkx():
    G0 = nx.MultiDiGraph()  # 创建多重有向图
    filename = os.path.join(project_path, "graph.xlsx")
    pdobj = pd.read_excel(filename, sheet_name='Sheet1', header=0, encoding="utf8")
    G = nx.from_pandas_edgelist(pdobj[["subject", "relation", "object"]], "subject", "object", edge_attr="relation",
                                create_using=G0)
    nx.draw(G, with_labels=True, font_weight='bold')
    # # 二阶节点
    # setall = set()
    # for i1 in nx.all_neighbors(G, "主1"):
    #     tmposet = set([i1])
    #     tmprset = set(nx.all_neighbors(G, i1))
    #     setall |= tmprset | tmposet
    # print(nx.info(G, "主1"))
    # 某节点某类属性的边
    alledge = nx.get_edge_attributes(G, name="relation")
    for i1 in alledge:
        if alledge[i1] == "系8":
            print(i1)
    for i1 in nx.all_neighbors(G, "主1"):
        res = nx.edges(G, i1)
        # res = nx.common_neighbors(G, "主1", "主2")
        # print(2)
        # print(res)
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to Delphis".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    test_networkx()
    # test_neo4j()
    exit(0)
    main()
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
