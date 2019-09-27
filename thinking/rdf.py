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
import json
import jieba
from utils.log_tool import *
from utils.connect_mysql import MysqlDB
import pymysql
import neo4j
from py2neo import Graph, Node, Relationship
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import copy
import itertools

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


def main():
    test_obj = [
        {"sentence": "一个月内，小明体重增加 $2kg$，小华体重减少 $1kg$，小强体重无变化，写出他们这个月的体重增长值"},
        {
            "sentence": "$2001$ 年下列国家的商品进出口总额比上年的变化情况是：\ 美国减少 $6.4\%$，德国增长 $1.3\%$，\ 法国减少 $2.4\%$，英国减少 $3.5\%$，\ 意大利增长 $0.2\%$，中国增长 $7.5\%$ ，\ 写出这些国家 $2001$ 年商品进出口总额的增长率"},
        {"sentence": "高出海平面记为正 ，低于海平面记为负 ，若地图上 $A$ ，$B$ 两地的高度分别标记为 $4600$ 米和 $-200$ 米 ，你能说出它们的含义吗"},
    ]
    from harvesttext.harvesttext import HarvestText

    doc = test_obj[0]["sentence"]
    ht = HarvestText()
    sentences = ht.cut_sentences(doc)
    print(sentences)
    inv_index = ht.build_index(sentences)
    # 实体计数
    counts = ht.get_entity_counts(sentences, inv_index)
    print(pd.Series(counts).sort_values(ascending=False).head())
    # 三元组提取
    res = ht.triple_extraction(sentences[0].strip())
    print(res)

    discrete_list = ins.sentences2triples(test_obj)
    # print(discrete_list)


def test_neo4j(domain_list=["六年级数学上学期"], point_json={}):
    # 图库驱动
    t_graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4j"))
    # 获取文件
    filename = os.path.join(project_path, "graph_work.xlsx")
    domain = "六年级数学上学期"
    pdobj = pd.read_excel(filename, sheet_name='Sheet1', header=0)
    # print(domain_list)
    # print(pdobj["domain"])
    pdobj[["subject", "relation", "object"]] = pdobj[pdobj["domain"].isin(domain_list)][
        ["subject", "relation", "object"]]
    # filename = os.path.join(project_path, "property.xlsx")
    properpobj = pd.read_excel(filename, sheet_name='Sheet2', header=0, encoding="utf8")
    properpobj[["object", "property", "value"]] = properpobj[properpobj["domain"].isin(domain_list)][
        ["object", "property", "value"]]
    # 先清空
    t_graph.delete_all()
    nodejson = {}
    for i1 in properpobj.iterrows():
        a = Node(str(i1[1].property), name=i1[1].object,
                 property=i1[1].property + i1[1].value if i1[1].value == "错" else i1[1].property)
        nodejson[i1[1].object] = a
        t_graph.create(a)
    for i1 in pdobj.iterrows():
        r = Relationship(nodejson[i1[1].subject], i1[1].relation, nodejson[i1[1].object])
        t_graph.create(r)


class Node_tool(object):
    # 只适合 必有起始点的 有向图
    def __init__(self, filename, domain_list):
        self.ori_graph = None
        self._get_data(filename, domain_list)
        self._get_adjacency()
        self.get_local_nodes()

    def _get_data(self, filename, domain_list):
        self.pdobj = pd.read_excel(filename, sheet_name='Sheet1', header=0, encoding="utf8")
        self.properpobj = pd.read_excel(filename, sheet_name='Sheet2', header=0, encoding="utf8")
        self.pdobj[["subject", "relation", "object"]] = self.pdobj[self.pdobj["domain"].isin(domain_list)][
            ["subject", "relation", "object"]]
        G0 = nx.MultiDiGraph()  # 创建多重有向图
        self.ori_graph = nx.from_pandas_edgelist(self.pdobj[["subject", "relation", "object"]], "subject", "object",
                                                 edge_attr="relation", create_using=G0)

    def _get_adjacency(self):
        self.adj = nx.adjacency_matrix(self.ori_graph)

    def _get_point_chaper(self, nodeid):
        tmpnodeid = nodeid
        # 循环索引找id
        for i1 in range(self.lenth):
            for i2 in range(self.lenth):
                if tmpnodeid in self.row_list:
                    tmpnodeid = self.col_list
                    if self.node_propety[tmpnodeid] == "章节":
                        return tmpnodeid
        return None

    def get_local_nodes(self):
        # 1. 基本数据生成
        tmpdata = self.adj.tocoo()
        self.row_list = tmpdata.row
        self.col_list = tmpdata.col
        self.lenth = len(self.col_list)  # 三元组个数
        # 三元组个数
        self.relation_propety = [i1[1] for i1 in
                                 zip(self.pdobj["subject"], self.pdobj["relation"], self.pdobj["object"])]
        map_prop = {i1[0]: i1[1] for i1 in zip(self.properpobj["object"], self.properpobj["property"])}
        # 点的个数
        self.node_propety = [map_prop[i1] for i1 in self.ori_graph.nodes]
        # 2. 起止点 章节
        self.local_root_nodes = []
        self.local_last_nodes = []
        # 只有起点不在终点列表里，就是局域起点
        for i1 in self.row_list:
            if i1 not in self.col_list:
                self.local_root_nodes.append(i1)
        # lastchapter_list = []
        for i1 in self.col_list:
            if i1 not in self.row_list:
                self.local_last_nodes.append(i1)
                # chapernode = self._get_point_chaper(i1)
                # if chapernode is None:
                #     raise Exception("知识点不应该没有章")
                # else:
                #     lastchapter_list.append(chapernode)
        # lastchapter_list = list(set(lastchapter_list))
        print(self.local_root_nodes)
        print(self.local_last_nodes)
        print(self.ori_graph.nodes)

    def get_node_order_list(self, point):
        local_root_nodes = self.get_global_root_nodes()

        # 每个错点一幅图
        # for i1 in points_list:
        #     local_root_nodes = self.get_global_root_nodes()
        nodelist = []
        return nodelist

    def get_nodes_order_list(self, points_list=[]):
        # 1. 得出错点的章节json
        # 2. 遍历虚拟展开领路径+原有路径
        # 3. 生成所有 错点->局域根路径。
        # 0 n01个起点 n02个终点 n03个错误点
        step_net_list = []
        wait_nodes = copy.deepcopy(points_list)
        dealed_nodes = []
        while len(wait_nodes) > 0:
            insig = 0
            # 遍历待处理点 找到的最短路径
            buf_wait_nodes = copy.deepcopy(wait_nodes)
            for wait_node in buf_wait_nodes:
                buf_step_net_list = copy.deepcopy(step_net_list)
                for stepid, stepinfo in enumerate(buf_step_net_list):
                    if wait_node in stepinfo[1]:
                        # 3. （拆枝）如存在于已有的路径列表 删除待处理列表的该项 拆分路径(新增点最多这可能拆一条路径)
                        wait_nodes.remove(wait_node)
                        dealed_nodes.append(wait_node)
                        step_net_list[stepid][0] = [stepinfo[0][0], wait_node]
                        step_net_list[stepid][1] = stepinfo[1][:stepinfo[0][1].index(wait_node)]
                        step_net_list.append(
                            [[wait_node, stepinfo[0][1]], stepinfo[1][stepinfo[0][1].index(wait_node):]])
                        break
            if 0 == insig:
                buf_wait_nodes = copy.deepcopy(wait_nodes)
                for wait_node in buf_wait_nodes:
                    # 1. （侧枝）尝试加入已处理有路径，成功一个就做拆分循环
                    deal_wait_node_sig = 0
                    tmpdeal_nodes = copy.deepcopy(dealed_nodes)
                    for tmpnode in tmpdeal_nodes:
                        # 遍历每一个已处理节点的路径
                        try:
                            sourcenode = tmpnode
                            targetnode = wait_node
                            path_t = nx.dijkstra_path(self.ori_graph, source=sourcenode, target=targetnode)
                            # 如果新增path中的非起止点包含 已处理点，舍弃这条路径
                            strippath = copy.deepcopy(path_t)
                            del strippath[-1]
                            del strippath[0]
                            usepath_sig = 1
                            for mid_node in strippath:
                                if mid_node in dealed_nodes:
                                    usepath_sig = 0
                                    break
                            if 1 == usepath_sig:
                                step_net_list.append([[sourcenode, targetnode], path_t])
                                dealed_nodes.append(tmpnode)
                                deal_wait_node_sig = 1
                        except Exception as e:
                            pass
                        try:
                            sourcenode = wait_node
                            targetnode = tmpnode
                            path_t = nx.dijkstra_path(self.ori_graph, source=sourcenode, target=targetnode)
                            # 如果新增path中的非起止点包含 已处理点，舍弃这条路径
                            strippath = copy.deepcopy(path_t)
                            del strippath[-1]
                            del strippath[0]
                            usepath_sig = 1
                            for mid_node in strippath:
                                if mid_node in dealed_nodes:
                                    usepath_sig = 0
                                    break
                            if 1 == usepath_sig:
                                step_net_list.append([[sourcenode, targetnode], path_t])
                                dealed_nodes.append(tmpnode)
                                deal_wait_node_sig = 1
                        except Exception as e:
                            pass
                    # 2. （新树）不存在则遍历已有的节点求最短路径
                    if deal_wait_node_sig == 1:
                        # 3. 判断成功跳出
                        break
                    else:
                        dealed_nodes.append(wait_node)
                        wait_nodes.remove(wait_node)
                        # linked_nodes = list(set(list(itertools.chain(*[i1[0] for i1 in step_net_list]))))
        # 4. 遍历每一条错点路径，删除存在路径中的，遍历未存在的。直到不存在可归并的。
        return step_net_list

    def show_origin_graph(self):
        self.properpobj.loc[self.properpobj["property"] == "非知识点", "color"] = '#ff0000'
        self.properpobj.loc[self.properpobj["property"] != "非知识点", "color"] = '#0000ff'
        self.properjson = self.properpobj[["object", "value"]].to_json(orient='records', force_ascii=False)
        val_map = {i1["object"]: i1["value"] for i1 in json.loads(self.properjson)}

        def map_node_colors(a):
            res0 = self.properpobj[(self.properpobj["object"] == a) & (self.properpobj["value"] == "错")]
            if len(res0) != 0:
                return "#ff0000"
            res1 = self.properpobj[(self.properpobj["object"] == a) & (self.properpobj["property"] == "非知识点")]
            if len(res1) != 0:
                return "#00ff00"
            res2 = self.properpobj[(self.properpobj["object"] == a) & (self.properpobj["property"] == "基础知识")]
            if len(res2) != 0:
                return "#000f0f"
            res3 = self.properpobj[(self.properpobj["object"] == a) & (self.properpobj["property"] == "基础技能")]
            if len(res3) != 0:
                return "#0f000f"
            res4 = self.properpobj[(self.properpobj["object"] == a) & (self.properpobj["property"] == "逻辑思维")]
            if len(res4) != 0:
                return "#0f0f00"

        node_colors = [map_node_colors(node) for node in self.ori_graph.nodes()]
        # edge
        self.pdobj.loc[self.pdobj["relation"] == "在教材的", "color"] = '#ff0000'
        self.pdobj.loc[self.pdobj["relation"] == "是成员", "color"] = '#00ff00'
        self.pdobj.loc[self.pdobj["relation"] == "是基础", "color"] = '#0000ff'

        def map_colors(a, b):
            res = self.pdobj[(self.pdobj["subject"] == a) & (self.pdobj["object"] == b)]
            return res["color"].values[0]

        edge_colors = [map_colors(a, b) for a, b in self.ori_graph.edges()]
        pos = nx.shell_layout(self.ori_graph)
        # pos = nx.random_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.circular_layout(G)
        nx.draw(self.ori_graph, pos, node_color=node_colors, edge_color=edge_colors, width=2, node_size=300,
                font_size=10,
                with_labels=True)
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        plt.show()


def test_networkx(domain_list=["六年级数学上学期"]):
    # 分析domain_list范围的知识点路径
    filename = os.path.join(project_path, "graph_work.xlsx")
    insnode = Node_tool(filename, domain_list)
    # 展示原始图
    insnode.show_origin_graph()
    # 生成错点关系 [[[错基1, 错展1],[path1]],[[错基2, 错展2],[path2]]]
    path_list = insnode.get_nodes_order_list(points_list=["分数的加减混合运算"])
    # nx.draw(G, with_labels=True, font_weight='bold')
    # # 二阶节点
    # setall = set()
    # for i1 in nx.all_neighbors(G, "主1"):
    #     tmposet = set([i1])
    #     tmprset = set(nx.all_neighbors(G, i1))
    #     setall |= tmprset | tmposet
    # print(nx.info(G, "主1"))
    # # 某节点某类属性的边
    # alledge = nx.get_edge_attributes(G, name="relation")
    # for i1 in alledge:
    #     if alledge[i1] == "系8":
    #         print(i1)
    # for i1 in nx.all_neighbors(G, "主1"):
    #     res = nx.edges(G, i1)
    #     # res = nx.common_neighbors(G, "主1", "主2")
    #     # print(2)
    #     # print(res)
    # nx.draw(G, pos=nx.random_layout(G), node_color='b', edge_color='r', with_labels=True, font_size=18, node_size=20)
    # lists = [("无理数", "实数", 500), ('有理数', "实数", 3.0)]
    # G.add_weighted_edges_from(lists)
    # node


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to Delphis".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    # test_networkx()
    # exit(0)
    # test_neo4j()
    main()
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
