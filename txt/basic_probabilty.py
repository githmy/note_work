from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time
from datetime import datetime
import numpy as np
import statsmodels.stats as stats
from scipy import stats
import pandas as pd


def probability():
    # 1. 二项式分布概率
    # 100次投币，正面朝上的次数， 20个样本
    res = np.random.binomial(100, 0.5, 20)
    print(res)
    # 100次投币，20次正面朝上的概率
    res = stats.binom.pmf(20, 100, 0.5)
    res = stats.binom.pmf([19, 20], 100, 0.5)
    print(res)
    # 100次投币，小于20次正面朝上的概率(积分)
    res = stats.binom.cdf(20, 100, 0.5)
    print(res)

    # 2. 正态分布概率
    # 正态分布概率， 5个样本随机数
    res = np.random.normal(size=5)
    print(res)
    # 随机数为 n的概率密度
    res = stats.norm.pdf([20, 100])
    print(res)
    # 随机数为 n的概率积分值
    res = stats.norm.cdf([20, 100])
    print(res)
    # 积分值为0.05的横坐标(分数位)
    mean = 0.5
    stand_var = 2 ** 0.5
    res = stats.norm.ppf(0.05, mean, stand_var)
    print(res)

    # 3. 卡方分布 z1^2+z2^2+z3^2 n个标准分布，均值为 n 方差 为 2*n
    # 随机数为 n的概率密度
    res = stats.chi.pdf([0.2, 0.3, 0.4], 3)
    print(res)

    # 4. t分布 X= Z/(Y/n)^(1/2)  Z=N(0,1) Y=卡方(n)
    # 随机数为 n的概率密度
    x = np.arange(-4, 4, 0.002)
    # df = 5, 30
    res = stats.t.pdf(x, 5)
    res = stats.t.pdf(x, 30)
    print(res)

    # 5. F(m,n)分布 X= (Z/m)/(Y/n)  Z=卡方(m) Y=卡方(n)
    # 随机数为 n的概率密度
    x = np.arange(0, 5, 0.002)
    # m=4, n=40
    res = stats.f.pdf(x, 4, 40)
    print(res)

def corr():
    # 1. 计算两列的相关系数
    pddata = pd.read_csv()
    corr_num = pddata["a"].corr(pddata["b"])
    print(corr_num)


if __name__ == '__main__':
    # 1. 时间计时秒 时间戳 1547387938
    timsteemp = time.time()
    print(timsteemp)
