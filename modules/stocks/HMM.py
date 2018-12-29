# -*- coding: utf-8 -*-
# from __future__ import unicode_literals
# from __future__ import print_function
# from __future__ import division
# from __future__ import absolute_import
#
import os
from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime

from scipy import stats  # To perform box-cox transformation
from sklearn import preprocessing  # To center and standardize the data.

# Source Code from previous HMM modeling

# Note that numbers of hidden states are modified to be 3, instead of 6.

# 测试时间从2005年1月1日到2015年12月31日，拿到每日沪深300的各种交易数据。
beginDate = '2005-01-01'
endDate = '2015-12-31'
n = 6  # Hidden states are set to be 3 instead of 6
typedict = {
    'open': np.float64,
    'high': np.float64,
    'close': np.float64,
    'low': np.float64,
    'volume': np.float64,
    'price_change': np.float64,
    'p_change': np.float64,
    'ma5': np.float64,
    'ma10': np.float64,
    'ma20': np.float64,
    'v_ma5': np.float64,
    'v_ma10': np.float64,
    'v_ma20': np.float64,
    'turnover': np.float64
}
cmd_path = os.getcwd()
cmd_path = os.path.join(cmd_path, "..")
data_path = os.path.join(cmd_path, "data")
data_path = os.path.join(data_path, "stock")
tmpfile = os.path.join(data_path, "000001_D.csv")
data = pd.read_csv(tmpfile, header=0, encoding="utf8", parse_dates=[0], index_col=0, dtype=typedict)
# data = get_price('CSI300.INDX', start_date=beginDate, end_date=endDate, frequency='1d')
# print(data[0:9])

# 拿到每日成交量和收盘价的数据。
volume = data['volume']
close = data['close']

# 计算每日最高最低价格的对数差值，作为特征状态的一个指标。
logDel = np.log(np.array(data['high'])) - np.log(np.array(data['low']))
# print(logDel)

# 计算每5日的指数对数收益差，作为特征状态的一个指标。
logRet_1 = np.array(np.diff(np.log(close)))  # 这个作为后面计算收益使用
logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
# print(logRet_5)

# 计算每5日的指数成交量的对数差，作为特征状态的一个指标。
logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))
# print(logVol_5)

# 由于计算中出现了以5天为单位的计算，所以要调整特征指标的长度。
logDel = logDel[5:]
logRet_1 = logRet_1[4:]
close = close[5:]
Date = pd.to_datetime(data.index[5:])

# 把我们的特征状态合并在一起。
A = np.column_stack([logDel, logRet_5, logVol_5])
# print(A)

# 下面运用 hmmlearn 这个包中的 GaussianHMM 进行预测
model = GaussianHMM(n_components=n, covariance_type="full", n_iter=2000)
# model.fit([A])
model.fit(A)
hidden_states = model.predict(A)
print(hidden_states)

# 我们把每个预测的状态用不同颜色标注在指数曲线上看一下结果。
plt.figure(figsize=(25, 18))
for i in range(model.n_components):
    pos = (hidden_states == i)
    plt.plot_date(Date[pos], close[pos], 'o', label='hidden state %d' % i, lw=2)
    plt.legend(loc="left")

# 从图中可以比较明显的看出绿色的隐藏状态代表指数大幅上涨，浅蓝色和黄色的隐藏状态代表指数下跌。
# 为了更直观的表现不同的隐藏状态分别对应了什么，我们采取获得隐藏状态结果后第二天进行买入的操作，这样可以看出每种隐藏状态代表了什么。
res = pd.DataFrame({'Date': Date, 'logRet_1': logRet_1, 'state': hidden_states}).set_index('Date')
plt.figure(figsize=(25, 18))
for i in range(model.n_components):
    pos = (hidden_states == i)
    pos = np.append(0, pos[:-1])  # 第二天进行买入操作
    df = res.logRet_1
    res['state_ret%s' % i] = df.multiply(pos)
    plt.plot_date(Date, np.exp(res['state_ret%s' % i].cumsum()), '-', label='hidden state %d' % i)
    plt.legend(loc="left")

# 可以看到，隐藏状态1是一个明显的大牛市阶段，隐藏状态0是一个缓慢上涨的阶段(可能对应反弹)，隐藏状态3和5可以分别对应震荡下跌的大幅下跌。
# 其他的两个隐藏状态并不是很明确。由于股指期货可以做空，我们可以进行如下操作：当处于状态0和1时第二天做多，当处于状态3和5第二天做空，其余状态则不持有。
long = (hidden_states == 0) + (hidden_states == 1)  # 做多
short = (hidden_states == 3) + (hidden_states == 5)  # 做空
long = np.append(0, long[:-1])  # 第二天才能操作
short = np.append(0, short[:-1])  # 第二天才能操作

# 收益曲线图如下：
res['ret'] = df.multiply(long) - df.multiply(short)
plt.plot_date(Date, np.exp(res['ret'].cumsum()), 'r-')

# 2. 通过直方图来观察所选指标（可观测序列）分布的正态性。其中第一个指标（每日最高最低价格的对数差值）明显偏离正态分布。
# the histogram of the raw observation sequences
n, bins, patches = plt.hist(logDel, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
n, bins, patches = plt.hist(logRet_5, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
n, bins, patches = plt.hist(logVol_5, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()

# 通过Box-Cox变换对第一个可观测序列进行调整，使其更接近正态分布。
# 同时对三个可观测序列进行标准化（调整期均值为0，且标准差调整为1个单位），保证可观测序列在参数估计中有大致相等的权重。
# Box-Cox Transformation of the observation sequences
boxcox_logDel, _ = stats.boxcox(logDel)

# Standardize the observation sequence distribution
rescaled_boxcox_logDel = preprocessing.scale(boxcox_logDel, axis=0, with_mean=True, with_std=True, copy=False)
rescaled_logRet_5 = preprocessing.scale(logRet_5, axis=0, with_mean=True, with_std=True, copy=False)
rescaled_logVol_5 = preprocessing.scale(logVol_5, axis=0, with_mean=True, with_std=True, copy=False)

# the histogram of the rescaled observation sequences
n, bins, patches = plt.hist(rescaled_boxcox_logDel, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
n, bins, patches = plt.hist(rescaled_logRet_5, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
n, bins, patches = plt.hist(rescaled_logVol_5, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()

# 把可观测序列组合成矩阵。
# Observation sequences matrix
A = np.column_stack([logDel, logRet_5, logVol_5])
# Rescaled observation sequences matrix
rescaled_A = np.column_stack([rescaled_boxcox_logDel, rescaled_logRet_5, rescaled_logVol_5])

# 对于未修正的可观测序列进行隐马尔科夫链建模。
# HMM modeling based on raw observation sequences

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=2000).fit([A])
hidden_states = model.predict(A)
hidden_states

# Plot the hidden states
plt.figure(figsize=(25, 18))
for i in range(model.n_components):
    pos = (hidden_states == i)
    plt.plot_date(Date[pos], close[pos], 'o', label='hidden state %d' % i, lw=2)
    plt.legend(loc="left")

# Trading test according to the hidden states
for i in range(3):
    pos = (hidden_states == i)
    pos = np.append(0, pos[:-1])  # 第二天进行买入操作
    df = res.logRet_1
    res['state_ret%s' % i] = df.multiply(pos)
    plt.plot_date(Date, np.exp(res['state_ret%s' % i].cumsum()), '-', label='hidden state %d' % i)
    plt.legend(loc="left")

# Trading test2 according to the hidden states
long = (hidden_states == 0)  # 做多
short = (hidden_states == 1)  # 做空
long = np.append(0, long[:-1])  # 第二天才能操作
short = np.append(0, short[:-1])  # 第二天才能操作

# Yield Curve
res['ret'] = df.multiply(long) - df.multiply(short)
plt.plot_date(Date, np.exp(res['ret'].cumsum()), 'r-')

# 对于修正的可观测序列进行隐马尔科夫链建模。
# HMM modeling based on processed observation sequences

rescaled_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=2000).fit([rescaled_A])
rescaled_hidden_states = rescaled_model.predict(rescaled_A)
rescaled_hidden_states

# Plot the hidden states
plt.figure(figsize=(25, 18))
for i in range(model.n_components):
    pos = (rescaled_hidden_states == i)
    plt.plot_date(Date[pos], close[pos], 'o', label='hidden state %d' % i, lw=2)
    plt.legend(loc="left")

# Trading test according to the hidden states
for i in range(3):
    pos = (rescaled_hidden_states == i)
    pos = np.append(0, pos[:-1])  # 第二天进行买入操作
    df = res.logRet_1
    res['state_ret%s' % i] = df.multiply(pos)
    plt.plot_date(Date, np.exp(res['state_ret%s' % i].cumsum()), '-', label='hidden state %d' % i)
    plt.legend(loc="left")

# Trading test2 according to the hidden states
long = (rescaled_hidden_states == 0)  # 做多
short = (rescaled_hidden_states == 1) + (rescaled_hidden_states == 2)  # 做空
long = np.append(0, long[:-1])  # 第二天才能操作
short = np.append(0, short[:-1])  # 第二天才能操作

# Yield Curve
res['ret'] = df.multiply(long) - df.multiply(short)
plt.plot_date(Date, np.exp(res['ret'].cumsum()), 'r-')
