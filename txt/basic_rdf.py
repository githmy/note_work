import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import neonx

# networkx 2 neo4j
# ! pip install neonx
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
import neo4j


def test_neo4j(tx):
    "start n=node(*)  match (n)-[r:observer]-()  delete n,r match (o:org{name: 'juxinli'}) match (n)-[r:observer]-()  delete o,r"
    tx.run("match (n) detach delete n")
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
        session.write_transaction(add_friend, "Arthur", "Guinevere")
        session.write_transaction(add_friend, "Arthur", "Lancelot")
        session.write_transaction(add_friend, "Arthur", "Merlin")
        session.read_transaction(print_friends, "Arthur")
        session.write_transaction(add_obj, "ab")
        # 1. 清空原有
        session.write_transaction(delete_all)


def test_networkx(fall_list=["数的整除", "比和比例"]):
    G0 = nx.MultiDiGraph()  # 创建多重有向图
    filename = os.path.join(project_path, "graph_work.xlsx")
    pdobj = pd.read_excel(filename, sheet_name='Sheet1', header=0, encoding="utf8")
    filename = os.path.join(project_path, "property.xlsx")
    properpobj = pd.read_excel(filename, sheet_name='Sheet1', header=0, encoding="utf8")
    pdobj[["subject", "relation", "object"]] = pdobj[pdobj["domain"] == "六年级数学上学期"][["subject", "relation", "object"]]
    G = nx.from_pandas_edgelist(pdobj[["subject", "relation", "object"]], "subject", "object", edge_attr="relation",
                                create_using=G0)
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
    properpobj.loc[properpobj["property"] == "非知识点", "value"] = '#ff0000'
    properpobj.loc[properpobj["property"] != "非知识点", "value"] = '#0000ff'
    properjson = properpobj[["object", "value"]].to_json(orient='records', force_ascii=False)
    val_map = {i1["object"]: i1["value"] for i1 in json.loads(properjson)}
    node_colors = [val_map.get(node, "#000000") for node in G.nodes()]
    # edge
    pdobj.loc[pdobj["relation"] == "在教材的", "value"] = '#ff0000'
    pdobj.loc[pdobj["relation"] == "是成员", "value"] = '#00ff00'
    pdobj.loc[pdobj["relation"] == "是基础", "value"] = '#0000ff'

    def map_colors(a, b):
        res = pdobj[(pdobj["subject"] == a) & (pdobj["object"] == b)]
        return res["value"].values[0]

    edge_colors = [map_colors(a, b) for a, b in G.edges()]

    # width
    short_path = nx.dijkstra_path(G, source=fall_list[0], target=fall_list[1])
    print('节点 {} 到 {} 的路径：'.format(*fall_list), short_path)

    def map_wide(a, b):
        if a in short_path and b in short_path:
            return 4
        else:
            return 2

    wide_list = [map_wide(a, b) for a, b in G.edges()]

    # 节点大小
    def map_size(a):
        if a in short_path:
            return 600
        else:
            return 300

    size_list = [map_size(node) for node in G.nodes()]
    pos = nx.shell_layout(G)
    # pos = nx.random_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.circular_layout(G)
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, width=wide_list, node_size=size_list, font_size=10,
            with_labels=True)
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def networkx2neo4j():
    import neonx
    # create a Networkx graph
    # LINKS_TO is the relatioship name between the nodes
    data = neonx.get_geoff(graph, "LINKS_TO")
    import json
    import datetime

    class DateEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, datetime.date):
                return o.strftime('%Y-%m-%d')
            return json.JSONEncoder.default(self, o)

    data = neonx.get_geoff(graph, "LINKS_TO", DateEncoder())
    results = neonx.write_to_neo("http://localhost:7474/db/data/", graph, 'LINKS_TO')
    results = neonx.write_to_neo("http://localhost:7474/db/data/", graph, 'LINKS_TO', 'Person')


def 绘图方式():
    G = nx.MultiDiGraph()  # 创建多重有向图
    nx.draw(g)
    list = [('a', 'b', 5.0), ('b', 'c', 3.0), ('a', 'c', 1.0)]
    G.add_weighted_edges_from([(list)])
    val_map = {'主1': '#ff0000',
               '主2': '#ff00ff',
               '主3': '#00ff00'}
    # - circular_layout：节点在一个圆环上均匀分布
    # - random_layout：节点随机分布
    # - shell_layout：节点在同心圆上分布
    # - spring_layout： 用Fruchterman - Reingold算法排列节点
    # - spectral_layout：根据图的拉普拉斯特征向量排列节
    pos = nx.random_layout(G)
    pos = nx.kamada_kawai_layout(G)
    pos = nx.circular_layout(G)
    pos = nx.shell_layout(G)
    pos = nx.spring_layout(G)
    pos = nx.shell_layout(G)

    # - node_size: 指定节点的尺寸大小(默认是300，单位未知，就是上图中那么大的点)
    # - node_color: 指定节点的颜色 (默认是红色，可以用字符串简单标识颜色，例如'r'为红色，'b'为绿色等，具体可查看手册)，用“数据字典”赋值的时候必须对字典取值（.values()）后再赋值
    # - node_shape: 节点的形状（默认是圆形，用字符串'o'标识，具体可查看手册）
    # - alpha: 透明度 (默认是1.0，不透明，0为完全透明)
    # - width: 边的宽度 (默认为1.0)
    # - edge_color: 边的颜色(默认为黑色)
    # - style: 边的样式(默认为实现，可选： solid|dashed|dotted,dashdot)
    # - with_labels: 节点是否带标签（默认为True）
    # - font_size: 节点标签字体大小 (默认为12)
    # - font_color: 节点标签字体颜色（默认为黑色）
    # nx.draw(G, pos, node_color=values, edge_color='r', with_labels=True, font_size=18, node_size=300)
    def map_colors(a, b):
        res = pdobj[(pdobj["subject"] == a) & (pdobj["object"] == b)]
        return res["value"].values[0]

    node_colors = [val_map.get(node, "#000000") for node in G.nodes()]
    edge_colors = [map_colors(a, b) for a, b in G.edges()]
    # 画图
    wide_list = [map_wide(a, b) for a, b in G.edges()]
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, width=wide_list, with_labels=True, font_size=18,
            node_size=300)

    nx.draw_random(g)  # 点随机分布
    nx.draw_circular(g)  # 点的分布形成一个环
    nx.draw_spectral(g)
    #
    partition = community.best_partition(User)
    size = float(len(set(partition.values())))
    pos = nx.spectral_layout(G)
    count = 0.
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                      if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=50,
                               node_color=str(count / size))
    nx.draw_networkx_edge_labels(H, pos, edge_labels=labels)
    nx.draw_networkx_edges(User, pos, with_labels=True, alpha=0.5)


def 构建方式():
    # 从字典生成图
    dod = {0: {1: {'weight': 1}}}
    G = nx.from_dict_of_dicts(dod)  # 或G=nx.Graph(dpl)
    plt.subplots(1, 1, figsize=(6, 3))
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # 图转换为字典
    print(nx.to_dict_of_dicts(G))

    # 从列表中创建graph
    dol = {0: [1, 2, 3]}
    edgelist = [(0, 1), (0, 3), (2, 3)]
    G1 = nx.from_dict_of_lists(dol)  # 或G=nx.Graph(dol)
    G2 = nx.from_edgelist(edgelist)
    # 显示graph
    plt.subplots(1, 2, figsize=(15, 3))
    plt.subplot(121)
    nx.draw(G1, with_labels=True, font_weight='bold')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    nx.draw(G2, with_labels=True, font_weight='bold')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # graph转list
    print(nx.to_dict_of_lists(G1))
    print(nx.to_edgelist(G1))

    # 从numpy创建graph
    import numpy as np
    a = np.reshape(np.random.random_integers(0, 1, size=100), (10, 10))
    D = nx.DiGraph(a)
    nx.draw(D, with_labels=True, font_weight='bold')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # graph返回numpy
    G = nx.Graph()
    G.add_edge(1, 2, weight=7.0, cost=5)
    A1 = nx.to_numpy_matrix(G)
    A2 = nx.to_numpy_recarray(G, dtype=[('weight', float), ('cost', int)])
    print(A1, A2)

    # 从scipy创建graph
    G.clear()
    import scipy as sp
    A = sp.sparse.eye(2, 2, 1)
    G = nx.from_scipy_sparse_matrix(A)
    nx.draw(D, with_labels=True, font_weight='bold')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # graph返回scipy
    A = nx.to_scipy_sparse_matrix(G)
    print(A.todense())

    # 从pandas创建graph
    G.clear()
    import pandas as pd
    df = pd.DataFrame([[1, 1], [2, 1]])
    """
       weight  cost  0  b
    0       4     7  A  D
    1       7     1  B  A
    2      10     9  C  E
    """
    G = nx.from_pandas_edgelist(df, 0, 'b', ['weight', 'cost'], create_using=G0)
    G = nx.from_pandas_adjacency(df)
    nx.draw(D, with_labels=True, font_weight='bold')
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # graph返回scipy
    df = nx.to_pandas_adjacency(G)
    print(df)


def rdf_build():
    G = nx.Graph()  # 创建无向图
    G = nx.DiGraph()  # 创建有向图
    G = nx.MultiGraph()  # 创建多重无向图
    G = nx.MultiDiGraph()  # 创建多重有向图
    G.clear()  # 清空图

    G = nx.Graph()  # 建立一个空的无向图G
    G.add_node('a')  # 添加一个节点1
    G.add_nodes_from(['b', 'c', 'd', 'e'])  # 加点集合
    G.add_cycle(['f', 'g', 'h', 'j'])  # 加环
    H = nx.path_graph(10)  # 返回由10个节点挨个连接的无向图，所以有9条边
    G.add_nodes_from(H)  # 创建一个子图H加入G
    G.add_node(H)  # 直接将图作为节点

    nx.draw(G, with_labels=True)
    plt.show()

    # 图的属性
    G = nx.Graph(day='Monday')  # 可以在创建图时分配图的属性
    print(G.graph)
    G.graph['day'] = 'Friday'  # 也可以修改已有的属性
    print(G.graph)
    G.graph['name'] = 'time'  # 可以随时添加新的属性到图中
    print(G.graph)

    # 有向图转化成无向图
    H = DG.to_undirected()
    # 或者
    H = nx.Graph(DG)
    # 无向图转化成有向图
    F = H.to_directed()
    # 或者
    F = nx.DiGraph(H)
    nx.subgraph(G, nbunch)  # induce subgraph of G on nodes in nbunch
    nx.union(G1, G2)  # graph union
    nx.disjoint_union(G1, G2)  # graph union assuming all nodes are different
    nx.cartesian_product(G1, G2)  # return Cartesian product graph
    nx.compose(G1, G2)  # combine graphs identifying nodes common to both
    nx.complement(G)  # graph complement
    nx.create_empty_copy(G)  # return an empty copy of the same graph class
    nx.convert_to_undirected(G)  # return an undirected representation of G
    nx.convert_to_directed(G)  # return a directed representation of G

    # 访问节点
    print('图中所有的节点', G.nodes())
    print('图中节点的个数', G.number_of_nodes())

    # 删除节点
    G.remove_node(1)  # 删除指定节点
    G.remove_nodes_from(['b', 'c', 'd', 'e'])  # 删除集合中的节点
    nx.draw(G, with_labels=True)
    plt.show()

    # 节点的属性
    G = nx.Graph(day='Monday')
    G.add_node(1, index='1th')  # 在添加节点时分配节点属性
    print(G.node(data=True))
    G.node[1]['index'] = '0th'  # 通过G.node[][]来添加或修改属性
    print(G.node(data=True))
    G.add_nodes_from([2, 3], index='2/3th')  # 从集合中添加节点时分配属性
    print(G.nodes(data=True))
    print(G.node(data=True))

    # 添加边
    F = nx.Graph()  # 创建无向图
    F.add_edge(11, 12)  # 一次添加一条边
    g[1][2]['color'] = 'blue'
    g.add_edge(1, 2, weight=4.7)
    g.add_edges_from([(3, 4), (4, 5)], color='red')
    g.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
    g[1][2]['weight'] = 4.7
    g.edge[1][2]['weight'] = 4
    # 等价于
    e = (13, 14)  # e是一个元组
    F.add_edge(*e)  # 这是python中解包裹的过程
    F.add_edges_from([(1, 2), (1, 3)])  # 通过添加list来添加多条边
    # 通过添加任何ebunch来添加边
    F.add_edges_from(H.edges())  # 不能写作F.add_edges_from(H)
    e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
    G.add_weighted_edges_from(e)
    print(nx.dijkstra_path(G, 'a', 'd'))
    nx.draw(F, with_labels=True)
    plt.show()

    G = nx.DiGraph()
    G.add_edges_from([('n', 'n1'), ('n', 'n2'), ('n', 'n3')])
    G.add_edges_from([('n4', 'n41'), ('n1', 'n11'), ('n1', 'n12'), ('n1', 'n13')])
    G.add_edges_from([('n2', 'n21'), ('n2', 'n22')])
    G.add_edges_from([('n13', 'n131'), ('n22', 'n221')])
    G.add_edges_from([('n131', 'n221'), ('n221', 'n131')])
    G.add_node('n5')
    nx.draw(G, with_labels=True)
    plt.show()
    # 使用out_degree函数查找所有带有子项的节点：
    print([k for k, v in G.out_degree().iteritems() if v > 0])
    # 所有没有孩子的节点：
    print([k for k, v in G.out_degree().iteritems() if v == 0])
    # 所有孤儿节点，即度数为0的节点：
    print([k for k, v in G.degree().iteritems() if v == 0])
    # 超过2个孩子的节点
    print([k for k, v in G.out_degree().iteritems() if v > 2])

    # 访问边
    print('图中所有的边', F.edges())
    print('图中边的个数', F.number_of_edges())

    # 删除边
    F.remove_edge(1, 2)
    F.remove_edges_from([(11, 12), (13, 14)])
    nx.draw(F, with_labels=True)
    plt.show()

    # 边的属性
    G = nx.Graph(day='manday')
    G.add_edge(1, 2, weight=10)  # 在添加边时分配属性
    print(G.edges(data=True))
    G.add_edges_from([(1, 3), (4, 5)], len=22)  # 从集合中添加边时分配属性
    print(G.edges(data='len'))
    G.add_edges_from([(3, 4, {'hight': 10}), (1, 4, {'high': 'unknow'})])
    print(G.edges(data=True))
    G[1][2]['weight'] = 100000  # 通过G[][][]来添加或修改属性
    print(G.edges(data=True))

    # 快速遍历每一条边，可以使用邻接迭代器实现，对于无向图，每一条边相当于两条有向边
    FG = nx.Graph()
    FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.275)])
    for n, nbrs in FG.adjacency():
        for nbr, eattr in nbrs.items():
            data = eattr['weight']
            print('(%d, %d, %0.3f)' % (n, nbr, data))
    print('***********************************')
    # 筛选weight小于0.5的边：
    FG = nx.Graph()
    FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.275)])
    for n, nbrs in FG.adjacency():
        for nbr, eattr in nbrs.items():
            data = eattr['weight']
            if data < 0.5:
                print('(%d, %d, %0.3f)' % (n, nbr, data))
    print('***********************************')
    # 一种方便的访问所有边的方法:
    for u, v, d in FG.edges(data='weight'):
        print((u, v, d))

    """
    图 函数
    nx.degree(G[, nbunch, weight])：返回单个节点或nbunch节点的度数视图。
    nx.degree_histogram(G)  # 返回每个度值的频率列表。（从1至最大度的出现频次）
    nx.density(G)：返回图的密度。
    # 节点度中心系数。通过节点的度表示节点在图中的重要性，默认情况下会进行归一化，其值表达为节点度d(u)除以n-1（其中n-1就是归一化使用的常量）。这里由于可能存在循环，所以该值可能大于1.
    nx.degree_centrality(G) 
    # {1: 0.5, 2: 0.5, 3: 0.75, 4: 0.5, 5: 0.25}
    # 节点距离中心系数。通过距离来表示节点在图中的重要性，一般是指节点到其他节点的平均路径的倒数，这里还乘以了n-1。该值越大表示节点到其他节点的距离越近，即中心性越高。
    nx.closeness_centrality(G)  
    # {1: 0.5714285714285714, 2: 0.5714285714285714, 3: 0.80000000000000004, 4: 0.66666666666666663, 5: 0.44444444444444442}
    # 节点介数中心系数。在无向图中，该值表示为节点作占最短路径的个数除以((n-1)(n-2)/2)；在有向图中，该值表达为节点作占最短路径个数除以((n-1)(n-2))。
    nx.betweenness_centrality(G)  
    # {1: 0.0, 2: 0.0, 3: 0.66666666666666663, 4: 0.5, 5: 0.0}
    nx.triangles(G)
    # {1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
    # 图或网络的传递性。即图或网络中，认识同一个节点的两个节点也可能认识双方，计算公式为3*图中三角形的个数/三元组个数（该三元组个数是有公共顶点的边对数，这样就好数了）。
    nx.transitivity(G)
    # 0.5
    # 图或网络中节点的聚类系数。计算公式为：节点u的两个邻居节点间的边数除以((d(u)(d(u)-1)/2)
    nx.clustering(G)
    # {1: 1.0, 2: 1.0, 3: 0.33333333333333331, 4: 0.0, 5: 0.0}
    nx.info(G[, n])：打印图G或节点n的简短信息摘要。
    nx.create_empty_copy(G[, with_data])：返回图G删除所有的边的拷贝。
    nx.is_directed(G)：如果图是有向的，返回true。
    nx.add_star(G_to_add_to, nodes_for_star, **attr)：在图形G_to_add_to上添加一个星形。
    nx.add_path(G_to_add_to, nodes_for_path, **attr)：在图G_to_add_to中添加一条路径。
    nx.add_cycle(G_to_add_to, nodes_for_cycle, **attr)：向图形G_to_add_to添加一个循环。
    
    节点 函数
    nx.nodes(G)：在图节点上返回一个迭代器。
    nx.number_of_nodes(G)：返回图中节点的数量。
    nx.all_neighbors(graph, node)：返回图中节点的所有邻居。
    nx.non_neighbors(graph, node)：返回图中没有邻居的节点。
    nx.common_neighbors(G, u, v)：返回图中两个节点的公共邻居。
    
    边 函数
    nx.edges(G[, nbunch])：返回与nbunch中的节点相关的边的视图。只找相关的主体端边
    nx.number_of_edges(G)：返回图中边的数目。
    nx.non_edges(graph)：返回图中不存在的边。
    G.add_path([10,11,12])  #再来一个一字长蛇型网络，节点分别是10,11,12
    nx.has_path(G,source,target)
    """


def 最短路径():
    # dijkstra_path(G, source, target, weight='weight')             ————求最短路径
    # dijkstra_path_length(G, source, target, weight='weight')      ————求最短距离
    import networkx as nx
    import pylab
    import numpy as np
    # 自定义网络
    row = np.array([0, 0, 0, 1, 2, 3, 6])
    col = np.array([1, 2, 3, 4, 5, 6, 7])
    value = np.array([1, 2, 1, 8, 1, 3, 5])

    print('生成一个空的有向图')
    G = nx.DiGraph()
    print('为这个网络添加节点...')
    for i in range(0, np.size(col) + 1):
        G.add_node(i)
    print('在网络中添加带权中的边...')
    for i in range(np.size(row)):
        G.add_weighted_edges_from([(row[i], col[i], value[i])])

    print('给网路设置布局...')
    pos = nx.shell_layout(G)
    print('画出网络图像：')
    nx.draw(G, pos, with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5)
    pylab.title('Self_Define Net', fontsize=15)
    pylab.show()

    '''Shortest Path with dijkstra_path'''
    print('dijkstra方法寻找最短路径：')
    path = nx.dijkstra_path(G, source=0, target=7)
    print('节点0到7的路径：', path)
    print('dijkstra方法寻找最短距离：')
    distance = nx.dijkstra_path_length(G, source=0, target=7)
    print('节点0到7的距离为：', distance)


def 最小生成树():
    def prim(G, s):
        dist = {}  # dist记录到节点的最小距离
        parent = {}  # parent记录最小生成树的双亲表
        Q = list(G.nodes())  # Q包含所有未被生成树覆盖的节点
        MAXDIST = 9999.99  # MAXDIST表示正无穷，即两节点不邻接

        # 初始化数据
        # 所有节点的最小距离设为MAXDIST，父节点设为None
        for v in G.nodes():
            dist[v] = MAXDIST
            parent[v] = None
        # 到开始节点s的距离设为0
        dist[s] = 0

        # 不断从Q中取出“最近”的节点加入最小生成树
        # 当Q为空时停止循环，算法结束
        while Q:
            # 取出“最近”的节点u，把u加入最小生成树
            u = Q[0]
            for v in Q:
                if (dist[v] < dist[u]):
                    u = v
            Q.remove(u)

            # 更新u的邻接节点的最小距离
            for v in G.adj[u]:
                if (v in Q) and (G[u][v]['weight'] < dist[v]):
                    parent[v] = u
                    dist[v] = G[u][v]['weight']
        # 算法结束，以双亲表的形式返回最小生成树
        return parent

    import matplotlib.pyplot as plt
    import networkx as nx
    g_data = [(1, 2, 1.3), (1, 3, 2.1), (1, 4, 0.9), (1, 5, 0.7), (1, 6, 1.8), (1, 7, 2.0), (1, 8, 1.8), (2, 3, 0.9),
              (2, 4, 1.8), (2, 5, 1.2), (2, 6, 2.8), (2, 7, 2.3), (2, 8, 1.1), (3, 4, 2.6), (3, 5, 1.7), (3, 6, 2.5),
              (3, 7, 1.9), (3, 8, 1.0), (4, 5, 0.7), (4, 6, 1.6), (4, 7, 1.5), (4, 8, 0.9), (5, 6, 0.9), (5, 7, 1.1),
              (5, 8, 0.8), (6, 7, 0.6), (6, 8, 1.0), (7, 8, 0.5)]

    def draw(g):
        pos = nx.spring_layout(g)
        nx.draw(g, pos,
                arrows=True,
                with_labels=True,
                nodelist=g.nodes(),
                style='dashed',
                edge_color='b',
                width=2,
                node_color='y',
                alpha=0.5)
        plt.show()

    g = nx.Graph()
    # 显示树
    tree = prim(g, 1)
    mtg = nx.Graph()
    mtg.add_edges_from(tree.items())
    mtg.remove_node(None)
    draw(mtg)


def 最大联通子图及联通子图规模排序():
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.path_graph(4)
    G.add_path([10, 11, 12])
    nx.draw(G, with_labels=True, label_size=1000, node_size=1000, font_size=20)
    plt.show()
    # [print(len(c)) for c in sorted(nx.connected_components(G),key=len,reverse=True)]
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        print(c)  # 看看返回来的是什么？结果是{0,1,2,3}
        print(type(c))  # 类型是set
        print(len(c))  # 长度分别是4和3（因为reverse=True，降序排列）
    # 高效找出最大的联通成分，其实就是sorted里面的No.1
    largest_components = max(nx.connected_components(G), key=len)
    # 找出最大联通成分，返回是一个set{0,1,2,3}
    print(largest_components)
    print(len(largest_components))  # 4


def GCN_format():
    # 链表稀疏矩阵 .tolil 快， dotok次之，正常矩阵 次之，压缩矩阵 .tocsc 慢。使用时仍然是matrix[:,:]，打印时是转化形式。
    # graph : defaultdict(<class 'list'>, {0: [633, 1862, 2582], 1: [2, 652, 654],
    # adj : (0, 1862)	1
    #       (0, 2582)	1
    #       (1, 2)	1
    #       (1, 652)	1
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # 生成的是二进制文件
    graph_filename1 = "a.bb"
    nx.write_adjlist(G1, graph_filename1)
    print(adj)
    """
    输出        紧邻对称矩阵 + I 的标准化   D个特征     卷积通道F
    Z[N,F] = RULE (A[N,N] * X[N,D] * W[D,F]) 
    """


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/
    # http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html
    最大联通子图及联通子图规模排序()
    # iris_demo()
