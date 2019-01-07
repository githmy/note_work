# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import os
import numpy as np
import tushare as ts
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num, datestr2num
from datetime import datetime

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False

cmd_path = os.getcwd()
data_pa = os.path.join(cmd_path, "data")
data_path_res = os.path.join(data_pa, "res")


def pandas_candlestick_ohlc(stock_data, otherseries=None):
    # 设置绘图参数，主要是坐标轴
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    dayFormatter = DateFormatter('%d')
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if (datetime.strptime(stock_data.index[0], "%Y-%m-%d") - datetime.strptime(stock_data.index[-1],
                                                                               "%Y-%m-%d")) < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
    ax.grid(True)
    # 创建K线图
    stock_array = np.array(stock_data.reset_index()[['date', 'open', 'high', 'low', 'close']])
    stock_array[:, 0] = datestr2num(stock_array[:, 0])
    candlestick_ohlc(ax, stock_array, colorup="red", colordown="green", width=0.4)
    # 可同时绘制其他折线图
    if otherseries is not None:
        for each in otherseries:
            plt.plot(stock_data[each], label=each)
        plt.legend()
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()


def plot_line_scatter_demo(x, y):
    # plt.plot(x, y, '--', lw=2)
    plt.scatter(x, y, s=[3, 100], c=["r", "#0F0F0F0F"])
    plt.xlabel('x1')
    plt.ylabel('y3')
    plt.title('Mercator: aaa')
    plt.grid(True)
    plt.show()


# plot_single(ts['2014-01-01':'2015-12-31'], 20, title='test_org')
# 股票均值
def plot_line_stock(ts, w, title='time_sequence'):
    roll_mean = ts.rolling(window=w).mean()
    roll_std = ts.rolling(window=w).std()
    pd_ewma = pd.ewma(ts, span=w)

    plt.clf()
    plt.figure()
    plt.grid()
    plt.plot(ts, color='blue', label='Original')
    plt.plot(roll_mean, color='red', label='Rolling Mean')
    plt.plot(roll_std, color='black', label='Rolling Std')
    plt.plot(pd_ewma, color='yellow', label='EWMA')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show()
    # plt.savefig('./PDF/' + title + '.pdf', format='pdf')


# 多只股票时间序列
def plot_timesq(datas):
    # plt.clf()
    plt.figure()
    plt.grid()
    colorlist = ["red", "blue", 'yellow', 'green', 'black']
    counter = 0
    for i1 in datas.columns:
        plt.plot(datas[i1], color=colorlist[counter % len(colorlist)], label=i1)
        counter += 1
    plt.legend(loc='best')
    plt.title('lines')
    plt.show()
    # plt.savefig('./PDF/' + title + '.pdf', format='pdf')


# 多只股票时间序列
def nplot_timesq(datas):
    # plt.clf()
    plt.figure()
    plt.grid()
    colorlist = ["red", "blue", 'yellow', 'green', 'black']
    counter = 0
    for i1 in datas.columns:
        plt.plot(datas[i1], color=colorlist[counter % len(colorlist)], label=i1)
        counter += 1
    plt.legend(loc='best')
    plt.title('lines')
    plt.show()
    # plt.savefig('./PDF/' + title + '.pdf', format='pdf')


# 相似曲线
def plot_similar(y, yhat, filename):
    y = y * 0.01
    yhat = yhat * 0.01
    allnum = np.append(y, yhat)
    ma = max(allnum)
    mi = min(allnum)
    # 长 宽 背景颜色
    plt.figure(figsize=(12, 6), facecolor='w')
    # plt.figure(facecolor='w')

    # 参数‘221’表示2(row)x2(colu),即将画布分成2x2，两行两列的4块区域，1表示选择图形输出的区域在第一块
    plt.subplot(221)
    plt.title(u'yhat-y similar', fontsize=20)
    plt.plot([mi, ma], [mi, ma], color='gray', linestyle=':', marker='o', lw=1)
    plt.xlim()
    plt.ylim()
    # plt.xlim((x1_min, x1_max))
    # plt.ylim((x2_min, x2_max))
    plt.scatter(yhat, y, s=[10], c=["#FF0000"], edgecolors='none')
    plt.xlabel(u'yhat axis', fontsize=16)
    plt.ylabel(u'y axis', fontsize=16)
    plt.legend(loc='best')
    plt.grid(b=True)

    zipd = zip(y, yhat)
    expect_array = [[(yi - yihat) / (yihat + 1), yi / (yihat + 1)] for yi, yihat in zipd if yihat > 0]
    expectnp = np.array(expect_array).T
    bias_expect = sum(expectnp[0]) / len(expectnp[0])
    ave_expect = sum(expectnp[1]) / len(expectnp[1])
    plt.subplot(222)
    plt.title(u'(y-yhat)/(yhat+1) >0 的偏离期望' + str(bias_expect).encode("utf8"), fontsize=20)
    for yi, yihat in zipd:
        plt.plot([yihat, yihat], [0, (yi - yihat) / (yihat + 1)], color='r', linestyle='-', marker='o', lw=2)
    plt.xlim()
    plt.ylim()
    plt.xlabel(u'yhat axis', fontsize=16)
    plt.ylabel(u'(y-yhat)/(yhat+1) axis', fontsize=16)
    plt.legend(loc='best')
    plt.grid(b=True)

    plt.subplot(223)
    plt.title(u'y/(yhat+1) >0 的期望' + str(ave_expect).encode("utf8"), fontsize=20)
    for yi, yihat in zipd:
        plt.plot([yihat, yihat], [0, yi / (yihat + 1)], color='r', linestyle='-', marker='o', lw=2)
    plt.xlim()
    plt.ylim()
    plt.xlabel(u'yhat axis', fontsize=16)
    plt.ylabel(u'y/(yhat+1) axis', fontsize=16)
    plt.legend(loc='best')
    plt.grid(b=True)

    # plt.savefig('./PDF/' + title + '.pdf', format='pdf')
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename)
    plt.show()


# 相似曲线
def pd_similar(pdobj, filename):
    ma_map = {i1: max(pdobj[[i1, "predict_" + i1]].max()) for i1 in pdobj.columns if i1.startswith("ylabel_")}
    mi_map = {i1: min(pdobj[[i1, "predict_" + i1]].min()) for i1 in pdobj.columns if i1.startswith("ylabel_")}

    pagenum = 4
    rown = 2
    coln = 2
    yindex = 0
    fig, axes = plt.subplots(rown, coln)
    for i1, i2 in enumerate(pdobj.columns):
        if i2.startswith("ylabel_"):
            if yindex > pagenum - 1:
                yindex = 0
                fig, axes = plt.subplots(rown, coln)
            rowi = yindex // coln
            coli = yindex % coln
            # print(yindex ,rowi,coli)
            expection = (pdobj["predict_" + i2] - pdobj[i2]).mean()
            convar = (pdobj["predict_" + i2] - pdobj[i2]).std()
            skew = (pdobj["predict_" + i2] - pdobj[i2]).skew()
            kurt = (pdobj["predict_" + i2] - pdobj[i2]).kurt()
            axes[rowi][coli].set_title("1:%.2f,2:%.2f,3:%.2f,4:%.2f" % (expection, convar, skew, kurt))
            # axes[0].set_ylabel(ylabell)
            # axes[rowi][coli].set_xlabel("predict_%s" % i2)
            # x,y分别设置x轴，y轴的列标签或列的位置
            ax0 = pdobj.plot(kind="scatter", ax=axes[rowi][coli], color='r', x="predict_%s" % i2, y=i2)
            ax0.plot([mi_map[i2], ma_map[i2]], [mi_map[i2], ma_map[i2]], color='gray', linestyle=':', marker='o', lw=1)
            ax0.grid(b=True)
            yindex += 1

    # ylabell = u'ylabel_p_change'
    # axes[0][0].set_title(ylabell)
    # # axes[0].set_ylabel(ylabell)
    # # axes[0].set_xlabel(xlabell)
    # ax0 = pdobj.plot(kind="scatter", ax=axes[0][0], color='r', x=xlabell, y=ylabell)  # x,y分别设置x轴，y轴的列标签或列的位置
    # ax0.plot([mi, ma], [mi, ma], color='gray', linestyle=':', marker='o', lw=1)
    # ax0.grid(b=True)

    xlabell = u'predict_ylabel_p_change'
    if yindex > pagenum - 1:
        yindex = 0
        fig, axes = plt.subplots(rown, coln)
    rowi = yindex // coln
    coli = yindex % coln
    yindex += 1
    # pdobj_positive = pdobj[["y/(yhat+1)", "(y-yhat)/(yhat+1)"]][pdobj["predict_ylabel_p_change"] > 0]
    # bias_expect = pdobj_positive["(y-yhat)/(yhat+1)"].sum()/pdobj_positive.shape[0]
    # ave_expect = pdobj_positive["y/(yhat+1)"].sum()/pdobj_positive.shape[0]
    ylabell = '(y-yhat)/(yhat+1)'
    # axes[rowi][coli].set_title(u'(y-yhat)/(yhat+1) >0 的期望' + str(bias_expect).encode("utf8"), fontsize=20)
    ax1 = pdobj.plot(kind="scatter", ax=axes[rowi][coli], color='r', x=xlabell, y=ylabell)
    ax1.grid(b=True)

    if yindex > pagenum - 1:
        yindex = 0
        fig, axes = plt.subplots(rown, coln)
    rowi = yindex // coln
    coli = yindex % coln
    yindex += 1
    ylabell = 'y/(yhat+1)'
    # axes[rowi][coli].set_title(u'y/(yhat+1) >0 的期望' + str(ave_expect).encode("utf8"), fontsize=20)
    ax2 = pdobj.plot(kind="scatter", ax=axes[rowi][coli], color='r', x=xlabell, y=ylabell)  # x,y分别设置x轴，y轴的列标签或列的位置
    ax2.grid(b=True)
    # ax = pdobj.plot(secondary_y=['A', 'B'])  # 设置2个列轴，分别对各个列轴画折线图。ax（axes）可以理解为子图，也可以理解成对黑板进行切分，每一个板块就是一个axes
    # ax.right_ax.set_ylabel('AB scale')
    # ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.25, ...)
    # ax.legend(loc=2)  # 设置图例的位置
    plt.legend(loc='best')
    # plt.grid(b=True)

    # plt.legend(loc=1)
    if os.path.isfile(os.path.join(data_path_res, filename)):
        os.remove(os.path.join(data_path_res, filename))
    plt.savefig(os.path.join(data_path_res, filename))
    plt.show()


# 相似曲线
def npd_similar(pdobj, filename):
    ma_map = {i1: max(pdobj[[i1, "predict_" + i1]].max()) for i1 in pdobj.columns if i1.startswith("ylabel_")}
    mi_map = {i1: min(pdobj[[i1, "predict_" + i1]].min()) for i1 in pdobj.columns if i1.startswith("ylabel_")}

    pagenum = 4
    rown = 2
    coln = 2
    yindex = 0
    fig, axes = plt.subplots(rown, coln)
    for i1, i2 in enumerate(pdobj.columns):
        if i2.startswith("ylabel_"):
            if yindex > pagenum - 1:
                yindex = 0
                fig, axes = plt.subplots(rown, coln)
            rowi = yindex // coln
            coli = yindex % coln
            # print(yindex ,rowi,coli)
            expection = (pdobj["predict_" + i2] - pdobj[i2]).mean()
            convar = (pdobj["predict_" + i2] - pdobj[i2]).std()
            skew = (pdobj["predict_" + i2] - pdobj[i2]).skew()
            kurt = (pdobj["predict_" + i2] - pdobj[i2]).kurt()
            axes[rowi][coli].set_title("1:%.2f,2:%.2f,3:%.2f,4:%.2f" % (expection, convar, skew, kurt))
            # axes[0].set_ylabel(ylabell)
            # axes[rowi][coli].set_xlabel("predict_%s" % i2)
            # x,y分别设置x轴，y轴的列标签或列的位置
            ax0 = pdobj.plot(kind="scatter", ax=axes[rowi][coli], color='r', x="predict_%s" % i2, y=i2)
            ax0.plot([mi_map[i2], ma_map[i2]], [mi_map[i2], ma_map[i2]], color='gray', linestyle=':', marker='o', lw=1)
            ax0.grid(b=True)
            yindex += 1

    # ylabell = u'ylabel_p_change'
    # axes[0][0].set_title(ylabell)
    # # axes[0].set_ylabel(ylabell)
    # # axes[0].set_xlabel(xlabell)
    # ax0 = pdobj.plot(kind="scatter", ax=axes[0][0], color='r', x=xlabell, y=ylabell)  # x,y分别设置x轴，y轴的列标签或列的位置
    # ax0.plot([mi, ma], [mi, ma], color='gray', linestyle=':', marker='o', lw=1)
    # ax0.grid(b=True)

    xlabell = u'predict_ylabel_p_change'
    if yindex > pagenum - 1:
        yindex = 0
        fig, axes = plt.subplots(rown, coln)
    rowi = yindex // coln
    coli = yindex % coln
    yindex += 1
    # pdobj_positive = pdobj[["y/(yhat+1)", "(y-yhat)/(yhat+1)"]][pdobj["predict_ylabel_p_change"] > 0]
    # bias_expect = pdobj_positive["(y-yhat)/(yhat+1)"].sum()/pdobj_positive.shape[0]
    # ave_expect = pdobj_positive["y/(yhat+1)"].sum()/pdobj_positive.shape[0]
    ylabell = '(y-yhat)/(yhat+1)'
    # axes[rowi][coli].set_title(u'(y-yhat)/(yhat+1) >0 的期望' + str(bias_expect).encode("utf8"), fontsize=20)
    ax1 = pdobj.plot(kind="scatter", ax=axes[rowi][coli], color='r', x=xlabell, y=ylabell)
    ax1.grid(b=True)

    if yindex > pagenum - 1:
        yindex = 0
        fig, axes = plt.subplots(rown, coln)
    rowi = yindex // coln
    coli = yindex % coln
    yindex += 1
    ylabell = 'y/(yhat+1)'
    # axes[rowi][coli].set_title(u'y/(yhat+1) >0 的期望' + str(ave_expect).encode("utf8"), fontsize=20)
    ax2 = pdobj.plot(kind="scatter", ax=axes[rowi][coli], color='r', x=xlabell, y=ylabell)  # x,y分别设置x轴，y轴的列标签或列的位置
    ax2.grid(b=True)
    # ax = pdobj.plot(secondary_y=['A', 'B'])  # 设置2个列轴，分别对各个列轴画折线图。ax（axes）可以理解为子图，也可以理解成对黑板进行切分，每一个板块就是一个axes
    # ax.right_ax.set_ylabel('AB scale')
    # ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.25, ...)
    # ax.legend(loc=2)  # 设置图例的位置
    plt.legend(loc='best')
    # plt.grid(b=True)

    # plt.legend(loc=1)
    if os.path.isfile(os.path.join(data_path_res, filename)):
        os.remove(os.path.join(data_path_res, filename))
    plt.savefig(os.path.join(data_path_res, filename))
    plt.show()


def dim3():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    dimx = 10
    dimy = 10
    x = np.arange(0, dimx, 1)
    y = np.arange(0, dimy, 1)

    x, y = np.meshgrid(y, x)

    m = np.arange(dimx * dimy).reshape(dimx, dimy)
    for i in range(dimx):
        m[i, :] = i

    tt = 1 / np.power(1000, 2 * m / 100)
    for i in range(dimy):
        tt[:, i] = tt[:, i] * i

    z = np.sin(tt)
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)  # 用取样点(x,y,z)去构建曲面
    plt.show()


# 排序显示密度
def sort_density():
    target_col = "target"
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df[target_col].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('Loyalty Score', fontsize=12)
    plt.show()


# 显示区间密度
def range_density():
    plt.figure(figsize=(12, 8))
    sns.distplot(train_df[target_col].values, bins=50, kde=False, color="red")
    # 核密度估计 + 统计柱状图
    sns.distplot(stock['Daily Return'].dropna(), bins=100)
    # 核密度估计
    sns.kdeplot(stock['Daily Return'].dropna())
    # 两支股票的皮尔森相关系数
    sns.jointplot(stock['Daily Return'], stock['Daily Return'], alpha=0.2)
    plt.title("Histogram of Loyalty score")
    plt.xlabel('Loyalty score', fontsize=12)
    plt.show()


# 不同特征数值 的 方差分布
def chara_diffval_std():
    plt.figure(figsize=(8, 4))
    sns.violinplot(x="feature_3", y=target_col, data=train_df)
    plt.title("Feature 3 distribution")
    plt.xlabel('Feature 3', fontsize=12)
    plt.ylabel('Loyalty score', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()

# 中位数切割，区间画方差
def middle_split_box():
    bins = np.nanpercentile(train_df["sum_merch_trans"], range(0, 101, 10))
    print(bins)
    train_df['binned_sum_merch_trans'] = pd.cut(train_df['sum_merch_trans'], bins)
    # cnt_srs = train_df.groupby("binned_sum_hist_trans")[target_col].mean()
    print(train_df['binned_sum_merch_trans'])

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="binned_sum_merch_trans", y=target_col, data=train_df, showfliers=False)
    plt.xticks(rotation='vertical')
    plt.xlabel('binned sum of new merchant transactions', fontsize=12)
    plt.ylabel('Loyalty score', fontsize=12)
    plt.title("Sum of New merchants transaction value (Binned) distribution")
    plt.show()

# 相关性热图
def heat_fun():
    heatmap.set_clim(-1, 1)
    # 单独
    rets = df.dropna()
    plt.figure(1)
    sns.heatmap(rets.corr(), annot=True)
    plt.show()

# 散点标注
def scatter_fun():
    plt.figure(2)
    plt.scatter(rets.mean(), rets.std())
    plt.xlabel('Excepted Return')
    plt.ylabel('Risk')
    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(label, xy=(x, y), xytext=(15, 15), textcoords='offset points',
                     arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3'))
    plt.draw()
    # plt.close(2)
    plt.show()


if __name__ == '__main__':
    df = ts.get_hist_data('600848')
    # 1. 画图测试
    plot_line_scatter_demo(df["open"], df["high"])
    # 2. 蜡烛图测试
    pandas_candlestick_ohlc(df)
