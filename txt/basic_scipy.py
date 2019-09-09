import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def matrix_format():
    # 链表稀疏矩阵 .tolil 快， dotok次之，正常矩阵 次之，压缩矩阵 .tocsc 慢。使用时仍然是matrix[:,:]，打印时是转化形式。
    features = sp.vstack((allx, tx)).tolil()
    features = sp.vstack((allx, tx)).dotok()
    features = sp.vstack((allx, tx))
    features = sp.vstack((allx, tx)).tocsc()
    # 分解矩阵的行列
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    print(features)


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/
    # http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html
    iris_demo()
