# @Time : 2019/4/22 10:54
# @Author : YingbinQiu
# @Site :
# @File : gene2.py

import time

import geatpy as ga
import numpy as np

# help(ga.crtbp)
# help(ga.crtip)
# help(ga.crtpp)
# help(ga.crtrp)
"""============================目标函数============================"""


def aim(x):  # 传入种群染色体矩阵解码后的基因表现型矩阵
    y = x * np.sin(10 * np.pi * x) + 2.0
    return y[:, 0:1]


start_time = time.time()  # 开始计时
"""============================变量设置============================"""
x1 = [0, 100]  # 自变量1的范围
x2 = [-100, 0]  # 自变量2的范围
b1 = np.ones((1, 2))  # 自变量1是否包含下界
b2 = np.ones((1, 2))  # 自变量2是否包含上界
# range  [[1, 2, 3],
#         [3, 4, 5]]
# borders  [[1, 0, 1],
#           [0, 1, 1]]
ranges = np.vstack([x1, x2]).T  # 生成自变量的范围矩阵
borders = np.vstack([b1, b2]).T  # 生成自变量的边界矩阵
codes = [1, 1]  # (0:binary | 1:gray)，默认采用二进制编码
precisions = [5, 2]  # 变量的精度
scales = [0, 0]  # 1为使用对数刻度，0为使用算术刻度 默认采用算术刻度
print(111111)
print(ranges)
print(borders)
"""========================遗传算法参数设置========================="""
NIND = 40  # 种群个体数目
MAXGEN = 2  # 最大遗传代数
GGAP = 0.9  # 代沟：说明子代与父代的重复率为0.1
"""=========================开始遗传算法进化========================"""
FieldD = ga.crtfld(ranges, borders, precisions, codes, scales)  # 调用函数创建区域描述器
print(FieldD)
Lind = np.sum(FieldD[0, :])  # 计算编码后的染色体长度
print(Lind)
Chrom = ga.crtbp(NIND, Lind)  # 根据区域描述器生成二进制种群
# crtip(创建整数型种群)
# crtpp(创建排列编码种群)
# crtrp(创建实数型种群)
variable = ga.bs2rv(Chrom, FieldD)  # 对初始种群进行解码
print(variable.shape)
ObjV = aim(variable)  # 计算初始种群个体的目标函数值
print(ObjV.shape)
pop_trace = (np.zeros((MAXGEN, 2)) * np.nan)  # 定义进化记录器，初始值为nan
ind_trace = (np.zeros((MAXGEN, Lind)) * np.nan)  # 定义种群最优个体记录器，记录每一代最优个体的染色体，初始值为nan
for gen in range(MAXGEN):
    print("ggg")
    FitnV = ga.ranking(-ObjV)  # 根据目标函数大小分配适应度值(由于遵循目标最小化约定，因此最大化问题要对目标函数值乘上-1)
    print(FitnV.shape)
    SelCh = ga.selecting('sus', Chrom, FitnV, GGAP)  # sus(随机抽样选择) rws(轮盘赌选择) tour(锦标赛选择)
    print(SelCh.shape)
    # recdis(离散重组) recint(中间重组) reclin(线性重组) xovdp(两点交叉) xovdprs(减少代理的两点交叉) xovmp(多点交叉) xovpm(部分匹配交叉) xovsh(洗牌交叉) xovshrs(减少代理的洗牌交叉) xovsp(单点交叉) xovsprs(减少代理的单点交叉)
    SelCh = ga.recombin('xovsp', SelCh, 0.7)  # 重组(采用单点交叉方式，交叉概率为0.7)
    print(SelCh.shape)
    # mut(简单离散变异算子) mutbga(实数值变异算子) mutbin(二进制变异算子) mutgau(高斯突变算子) mutint(整数值变异算子) mutpp(排列编码变异算子)
    SelCh = ga.mutbin(SelCh)
    print(SelCh.shape)
    variable = ga.bs2rv(SelCh, FieldD)  # 对育种种群进行解码(二进制转十进制) # 9. 染色体解码  bs2int(二进制 / 格雷码转整数) bs2rv(二进制 / 格雷码转实数)
    print(variable.shape)
    ObjVSel = aim(variable)  # 求育种个体的目标函数值
    print(ObjVSel.shape)
    [Chrom, ObjV] = ga.reins(Chrom, SelCh, 1, 1, 1, -ObjV, -ObjVSel, ObjV, ObjVSel)  # 重插入得到新一代种群
    print("cc")
    print(Chrom.shape)
    print(ObjV.shape)
    # 记录
    best_ind = np.argmax(ObjV)  # 计算当代最优个体的序号
    print(best_ind)
    pop_trace[gen, 0] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
    pop_trace[gen, 1] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
    ind_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的变量值
# 进化完成
end_time = time.time()  # 结束计时
"""============================输出结果及绘图================================"""
print('目标函数最大值：', np.max(pop_trace[:, 0]))  # 输出目标函数最大值
variable = ga.bs2rv(ind_trace, FieldD)  # 解码得到表现型
print('用时：', end_time - start_time)
# 10. 数据可视化
# sgaplot(单目标进化动态绘图函数)
# trcplot(单目标进化跟踪器绘图)
# frontplot(多目标优化帕累托前沿绘图函数)

# 11. 多目标相关
# awGA(适应性权重法多目标聚合函数)
# rwGA(随机权重法多目标聚合函数)
# ndomin(简单非支配排序)
# ndomindeb(Deb 非支配排序)
# ndominfast(快速非支配排序)
# upNDSet(更新帕累托最优集)

# 12. 模板相关
# sga_real_templet(单目标进化算法模板(实值编码))
# sga_code_templet(单目标进化算法模板(二进制/格雷编码))
# sga_permut_templet(单目标进化算法模板(排列编码))
# sga_new_real_templet(改进的单目标进化算法模板(实值编码))
# sga_new_code_templet(改进的单目标进化算法模板(二进制/格雷编码))
# sga_new_permut_templet(改进的单目标进化算法模板(排列编码))
# sga_mpc_real_templet(基于多种群竞争进化单目标编程模板(实值编码))
# sga_mps_real_templet(基于多种群独立进化单目标编程模板(实值编码))
# moea_awGA_templet(基于适应性权重法(awGA) 的多目标优化进化算法模板)
# moea_rwGA_templet(基于随机权重法(rwGA) 的多目标优化进化算法模板)
# moea_nsga2_templet(基于改进NSGA-Ⅱ 算法的多目标优化进化算法模板)
# moea_q_sorted_new_templet(改进的快速非支配排序法的多目标优化进化算法模板)
# moea_q_sorted_templet(基于快速非支配排序法的多目标优化进化算法模板)
