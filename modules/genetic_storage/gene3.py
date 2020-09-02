# @Time : 2019/4/22 10:54
# @Author : YingbinQiu
# @Site :
# @File : gene3.py

import time

import geatpy as ga  # 导入geatpy库
import matplotlib.pyplot as plt
import numpy as np

""" aimfuc.py """


# ZDT1
def ZDT1(Chrom, LegV):
    ObjV1 = Chrom[:, 0]
    gx = 1 + (9 / 29) * np.sum(Chrom[:, 1:30], 1)
    hx = 1 - np.sqrt(ObjV1 / gx)
    ObjV2 = gx * hx

    return [np.array([ObjV1, ObjV2]).T, LegV]


import numpy as np
import geatpy as ga  # 导入geatpy库

AIM_M = __import__('aimfuc')  # 获取函数接口地址
AIM_F = 'ZDT1'  # 目标函数名

"""============================变量设置============================"""
ranges = np.vstack([np.zeros((1, 30)), np.ones((1, 30))])  # 生成自变量的范围矩阵
borders = np.vstack([np.ones((1, 30)), np.ones((1, 30))])  # 生成自变量的边界矩阵
precisions = [4] * 30  # 自变量的编码精度
"""========================遗传算法参数设置========================="""
NIND = 25  # 种群规模
MAXGEN = 1000  # 最大遗传代数
GGAP = 1  # 代沟：子代与父代的重复率为(1-GGAP),由于后面使用NSGA2算法，因此该参数无用
selectStyle = 'tour'  # 遗传算法的选择方式
recombinStyle = 'xovdprs'  # 遗传算法的重组方式，设为两点交叉
recopt = 0.9  # 交叉概率
pm = 0.1  # 变异概率
SUBPOP = 1  # 设置种群数为1f
maxormin = 1  # 设置标记表明这是最小化目标
MAXSIZE = 1000  # 帕累托最优集最大个数
FieldDR = ga.crtfld(ranges, borders, precisions)  # 生成区域描述器
"""=======================调用进化算法模板进行种群进化==================="""
# 得到帕累托最优解集NDSet以及解集对应的目标函数值NDSetObjV
[ObjV, NDSet, NDSetObjV, times] = ga.moea_nsga2_templet(AIM_M, AIM_F, None, None, FieldDR, 'R', maxormin, MAXGEN,
                                                        MAXSIZE, NIND, SUBPOP, GGAP, selectStyle, recombinStyle, recopt,
                                                        pm, distribute=True, drawing=1)
