import networkx as nx
import pylab
import numpy as np

import pandas as pd
import os

# pdobj = pd.read_csv("a.txt", header=None, encoding="utf8")
# orilist = pdobj.iloc[:, 0].values
orilist = ['pcL_586ef274065b7e9d71429683.mp4', 'pcL_586ef04b065b7e9d71429681.mp4', 'pcL_57d7079cba53a54020ced907.mp4',
           'pcL_5b3dbcea6e9b222f824985ad.mp4', 'pcL_586eb825065b7e9d71429645.mp4', 'pcL_586f449c065b7e9d714296d1.mp4',
           'pcL_586ebdfa065b7e9d71429649.mp4', 'pcL_586cf28c065b7e9d7142946a.mp4', 'pcL_586efb4e065b7e9d7142968d.mp4',
           'pcL_57d7c413ba53a54020ced9d3.mp4', 'pcL_57d9aa30ba53a54020cedc0a.mp4', 'pcL_5c11dd97e8200039b3d454dc.mp4',
           'pcL_5c11dbbae8200039b3d454d7.mp4', 'pcL_586ed8b8065b7e9d71429665.mp4', 'pcL_5b3dd82c6e9b222f824985b5.mp4',
           'pcL_586d380b065b7e9d714294d6.mp4', 'pcL_5a795e8227e2c04e2d381620.mp4', 'pcL_586d11f6065b7e9d71429496.mp4',
           'pcL_586d09dc065b7e9d7142948a.mp4', 'pcL_586d06e5065b7e9d71429486.mp4', 'pcL_5870936b065b7e9d714297ed.mp4',
           'pcL_586d1587065b7e9d7142949c.mp4', 'pcL_5a090b047347fe08b2108690.mp4', 'pcL_586d0b24065b7e9d7142948c.mp4',
           'pcL_586d73f9065b7e9d71429518.mp4', 'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtr4gr.mp4',
           'zasfdasttt.mp4', 'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtr4gr.mp4', 'zasfdasttt.mp4',
           'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtr4gr.mp4', 'zasfdasttt.mp4', 'zasdfjkdjsffijdfpp.mp4',
           'zasdtg5y5th.mp4', 'zasfdasttt.mp4', 'zasdtr4gr.mp4', 'zasfdasttt.mp4', 'zasdtr4gr.mp4',
           'zasdfjkdjsffijdfpp.mp4', 'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtg5y5th.mp4',
           'zasdfjkdjsffijdfpp.mp4', 'zasdtg5y5th.mp4', 'zasfdasttt.mp4', 'zasdtr4gr.mp4']
oklist = []
f_dir = "/share/题目视频/下载/待检数学/"
tansidr = "/share/题目视频/下载/tmp/trans"
for root, dirs, files in os.walk(f_dir, topdown=True):
    for ondir in dirs:
        for fileed in os.listdir(os.path.join(f_dir, ondir)):
            fileedN = fileed.replace("pcM_", "pcL_")
            # fileedN = fileed.replace("pcL_", "pcM_")
            if fileedN in orilist:
                oklist.append(fileedN)
                strss = "cp -f {} {}".format(os.path.join(f_dir, ondir, fileed), tansidr)
                print(strss)
                # os.system(strss)
    break

notlist = [i1 for i1 in orilist if i1 not in oklist]
print(notlist)
print(len(notlist))


def test():
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

    '''
    Shortest Path with dijkstra_path
    '''
    print('dijkstra方法寻找最短路径：')
    path = nx.dijkstra_path(G, source=0, target=7)
    # [0, 3, 6, 7]
    print('节点0到7的路径：', path)
    path = nx.multi_source_dijkstra_path(G, {0, 7, 3})
    # {0: [0], 7: [7], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 1, 4], 6: [0, 3, 6], 5: [0, 2, 5]}
    # {0: [0], 4: [4], 7: [7], 1: [0, 1], 2: [0, 2], 3: [0, 3], 6: [0, 3, 6], 5: [0, 2, 5]}
    # {0: [0], 3: [3], 7: [7], 1: [0, 1], 2: [0, 2], 6: [3, 6], 4: [0, 1, 4], 5: [0, 2, 5]}
    print('节点0到7的路径：', path)
    print('dijkstra方法寻找最短距离：')
    distance = nx.dijkstra_path_length(G, source=0, target=7)
    print('节点0到7的距离为：', distance)

    path = nx.dijkstra_path(G, source=7, target=0)
    print('节点7到0的路径：', path)
    print('dijkstra方法寻找最短距离：')
    distance = nx.dijkstra_path_length(G, source=7, target=0)
    print('节点7到0的距离为：', distance)


if __name__ == '__main__':
    # print(set(['zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtr4gr.mp4', 'zasfdasttt.mp4', 'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtr4gr.mp4', 'zasfdasttt.mp4', 'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtr4gr.mp4', 'zasfdasttt.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtg5y5th.mp4', 'zasfdasttt.mp4', 'zasdtr4gr.mp4', 'zasfdasttt.mp4', 'zasdtr4gr.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtg5y5th.mp4', 'zasdfjkdjsffijdfpp.mp4', 'zasdtg5y5th.mp4', 'zasfdasttt.mp4', 'zasdtr4gr.mp4']))
    # exit()
    test()
