# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import os
import numpy as np

# 控制输入样式
np.set_printoptions(edgeitems=3, infstr='inf',
                    linewidth=75, nanstr='nan', precision=8,
                    suppress=False, threshold=1000, formatter=None)
np.set_printoptions(precision=None,
                    threshold=None,
                    edgeitems=None,
                    linewidth=None,
                    suppress=None,
                    nanstr=None,
                    infstr=None,
                    formatter=None)
# precision:输出结果保留精度的位数
# threshold:array数量的个数在小于threshold的时候不会被折叠
# edgeitems:在array已经被折叠后，开头和结尾都会显示edgeitems个数
# formatter:这个很有意思，像python3里面str.format(),就是可以对你的输出进行自定义的格式化
# 其他的暂时没用到
np.set_printoptions(suppress=True)  # 不用科学计数法

# # 数组初始化
# [np.nan,] * 4
# items_matrix = np.full((2, 3, 4), np.nan)
# 字符串初始化，它可以存储最多256个字符的10个字符串
# strArr = np.empty(10, dtype=np.str)

# # 属性
# print(a.ndim)   #数组的维数
# 3
# print(a.shape)  #数组每一维的大小
# (2, 2, 2)
# print(a.size)   #数组的元素数
# 8
# print(a.dtype)  #元素类型
# float64
# print(a.itemsize)  #每个元素所占的字节数

# # 生成数组
# arr = np.arange(10)
# np.linspace(1, 10, 20)
# np.zeros((3, 4))
# np.eye(3)
# np.ones((3, 4))


# # 添加行
# nparray = np.row_stack((nparray, np.transpose([otherlist])))

# # 插入数据
# tmp_np = np.insert(np.diff(data_list[i2][i2], n=1), 0, values=[np.nan], axis=0)

# # 做差值
# y = np.diff(x, n=1)

# # 数组的除法
# np.true_divide(a,b)

# 计算
a = []
ts = []
np.sin(a)
np.log(a)
np.exp(a)
np.sqrt(sum((a - ts) ** 2) / len(ts))
np.max(a)
np.floor(a)
np.dot(a, a)  ##矩阵乘法
a.max()
a.min()
a.sum()
a.sum(axis=0)  # 计算每一列（二维数组中类似于矩阵的列）的和

# 差值
np.ptp(grade)
# 等价
np.max(grade) - np.min(grade)

# 为空的 bool 列
np.isnan(test1)
delpreaftsig = np.logical_or(delpresig, delaftsig)
x = x( ~ isnan(x)); 更快的做法
print(ychara_list[-1][~delpreaftsig])

# numpy 转pandas
# dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
# values = numpy.zeros(20, dtype=dtype)
# index = ['Row'+str(i) for i in range(1, len(values)+1)]
#
# df = pandas.DataFrame(values, index=index)

# # pandas 转 numpy
# res= df.as_matrix()
# res= df.values
# res= np.array(df)

# # 拼接垂直水平
# np.vstack((a, b))
# np.hstack((a, b))
# tt = np.concatenate([X_t,X_te],axis=0)
# print(tt.shape)


# # 二进制未压缩保存
# arr = np.arange(10)
# np.save("../../result/note_work/nptest", arr)
# np.load("../../result/note_work/nptest.npy")

# # 复制深拷贝
# a.copy()

# # 转置
# a.transpose()
X = np.array([]).T

# 删除不要的维度
np.squeeze(x)

# 增加维度
np.expand_dims(x, -1)

# # 本征值
# import numpy.linalg as nplg
#
# a = np.array([[1, 0], [2, 3]])
# nplg.eig(a)

# # 数组排序 order
# a = []
# y_test = np.append(a, [1, 3, 2, 4], axis=0)
# x_test = np.append(a, [2, 3, 5, 7], axis=0)
# order = y_test.argsort(axis=0)
# # np.argsort(a, axis=-1, kind='quicksort', order=None)
# # b = np.sort(a, axis=1)  # 对a按每行中元素从小到大排序
# print(order)
# y_test = y_test[order]
# x_test = x_test[order]
# print(y_test)
# print(x_test)
# np.msort(c)

# # 位置
# np.argmax([[], []], axis=1)

# 数组内存
arr1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
arr1 = np.ones((3, 3))
# 数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会。
arr2 = np.array(arr1)
arr3 = np.asarray(arr1)

# 类型数值的上下限
np.iinfo(np.int8).min
np.iinfo(np.int8).max

# 中位数
a = np.array([[10., 7., 4.], [3., 2., 1.]])
np.nanpercentile(a, 50)

# 中位数
np.median(c)

# 类型转化
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32');
c = a.astype(np.float32)

# np.corrcoef()方法计算数据皮尔逊积矩相关系数
x = np.vstack((a, b, c))
r = np.corrcof(x)

# 等价 np.log(1+x)
np.log1p(x)
# 等价 np.exp(x)-1
np.expm1(x)

# 过滤修改
data[:, 1][data[:, 1] < 5] = 5  # 对第2列小于 5 的替换为5
