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
# from mpl_finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num, datestr2num
from datetime import datetime
# ! pip install statsmodels
import statsmodels as stats

# %matplotlib

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False

cmd_path = os.getcwd()
data_pa = os.path.join(cmd_path, "data")
data_path_res = os.path.join(data_pa, "res")

# 阻塞开关
plt.ion()
plt.ioff()


def 保存图片():
    import matplotlib.image
    import skimage
    skimage.io.imsave('/tmp/test.jpg', image)
    matplotlib.image.imsave('name0.png', image)
    plt.imsave('name.png', image)
    # Image.fromarray(image).save('WordCloud.png')


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


# 画图改坐标轴
def plot_curve(x, ys, titles):
    yins = [np.array(y) for y in ys]
    xin = np.arange(0, len(ys[0]))
    nums = len(ys)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (nums // 7 + 1)
    # 长 宽 背景颜色
    plt.figure(figsize=(12, 6), facecolor='w')
    # plt.figure(facecolor='w')
    for n in range(nums):
        plt.plot(xin, yins[n], color=colors[n], linestyle='-', linewidth=1.2, marker="", markersize=7,
                 markerfacecolor='b', markeredgecolor='g', label=titles[n])
        plt.legend(loc='upper right', frameon=False)
    # plt.plot(xin, yin, color='r', linestyle='-', linewidth=1.2, marker="*", markersize=7, markerfacecolor='b',
    #          markeredgecolor='g')
    plt.xlabel("x", verticalalignment="top")
    plt.ylabel("y", rotation=0, horizontalalignment="right")
    # xticks = ["今天", "周五", "周六", "周日", "周一"]
    # show_inte = 30
    show_inte = 7
    s_xin = [i1 for i1 in xin if i1 % show_inte == 0]
    s_x = [i1 for id1, i1 in enumerate(x) if id1 % show_inte == 0]
    plt.xticks(s_xin, s_x, rotation=90, fontsize=10)
    # plt.xticks(xin, x, rotation=90, fontsize=5)
    # yticks = np.arange(0, 500, 10)
    # plt.yticks(yticks)
    # plt.title(title)
    # plt.grid(b=True)
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
    # plt.ylim(-1.5, 1.5)
    # plt.xlabel('日期')
    # plt.ylabel('价格')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.grid(True, axis='both')
    # plt.show()
    # plt.savefig('./PDF/' + title + '.pdf', format='pdf')

    a = [0, 0, 0, 0]
    for i in Close:
        if (i > 2) & (i <= 3):
            a[0] += 1
        elif (i > 3) & (i <= 4):
            a[1] += 1
        elif (i > 4) & (i <= 5):
            a[2] += 1
        else:
            a[3] += 1
    plt.bar([2, 3, 4, 5], a)
    plt.bar(left=[2, 3, 4, 5], height=a, width=1.0, bottom=2.0)
    plt.title('中国银行收盘价分布柱状图')

    plt.bar(left=[2, 3, 4, 5], height=a, width=1.0, bottom=2.0, color='red', edgecolor='k')
    plt.title('中国银行收盘价分布柱状图')

    plt.barh([2, 3, 4, 5], a, height=1.0, color='red', edgecolor='k')
    plt.title('中国银行收盘价分布柱状图')

    plt.hist(Close, bins=12)
    plt.title('中国银行收盘价分布直方图')

    plt.hist(Close, range=(2.3, 5.5), orientation='horizontal', color='red', edgecolor='blue')
    plt.title('中国银行收盘价分布直方图')

    plt.pie(a, labels=('（2,3]', '(3,4]', '(4,5]', '(5,6]'), colors=('b', 'g', 'r', 'c'), shadow=True)
    plt.title('中国银行收盘价分布饼图')


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
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))  # 用取样点(x,y,z)去构建曲面
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
    # bins = [0, 10, 20, 30, 40, 50, 75, 10000]
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


def basic_plot():
    def qq_polt():
        # QQ图 观测与预测值之间的差异
        plt.figure(2)
        res = stats.probplot(train['sale'], plot=plt)
        plt.show()


# 分布数据可视化 - 散点图
# https://blog.csdn.net/qq_42554007/article/details/82625118
def sns_demo1():
    import scipy.stats as sci
    # 散点图 + 分布图
    rs = np.random.RandomState(2)
    df = pd.DataFrame(rs.randn(200, 2), columns=['A', 'B'])
    sns.jointplot(x=df['A'], y=df['B'],  # 设置xy轴，显示columns名称
                  data=df,  # 设置数据
                  color='b',  # 设置颜色
                  s=50, edgecolor='w', linewidth=1,  # 设置散点大小、边缘颜色及宽度(只针对scatter)
                  stat_func=sci.pearsonr,
                  kind='scatter',  # 设置类型：'scatter' 散点,'reg' ,'resid','kde' 密集,'hex' 六边
                  # stat_func=<function pearsonr>,
                  space=0.1,  # 设置散点图和布局图的间距
                  size=8,  # 图表大小(自动调整为正方形))
                  ratio=5,  # 散点图与布局图高度比，整型
                  marginal_kws=dict(bins=15, rug=True),  # 设置柱状图箱数，是否设置rug
                  )


# 可拆分绘制的散点图
def sns_demo2():
    # plot_joint() + ax_marg_x.hist() + ax_marg_y.hist()
    # 设置风格
    sns.set_style('white')
    # 导入数据
    tips = sns.load_dataset('tips')
    print(tips.head())

    # 创建一个绘图表格区域，设置好x,y对应数据
    g = sns.JointGrid(x='total_bill', y='tip', data=tips)

    g.plot_joint(plt.scatter, color='m', edgecolor='white')  # 设置框内图表，scatter
    # g = g.plot_joint(plt.scatter, color='g', s=40, edgecolor='white')  # 绘制散点图
    # g = g.plot_joint(sns.kdeplot, cmap = 'Reds_r')     #绘制密度图
    g.ax_marg_x.hist(tips['total_bill'], color='b', alpha=.6,
                     bins=np.arange(0, 60, 3))  # 设置x轴为直方图，注意bins是数组
    g.ax_marg_y.hist(tips['tip'], color='r', alpha=.6,
                     orientation='horizontal',
                     bins=np.arange(0, 12, 1))  # 设置x轴直方图，注意需要orientation参数
    from scipy import stats
    g.annotate(stats.pearsonr)
    # 设置标注，可以为pearsonar， spearmanr


# 矩阵散点图 - pairplot()
def sns_demo3():
    # 设置风格
    sns.set_style('white')
    # 读取数据
    iris = sns.load_dataset('iris')
    print(iris.head())
    sns.pairplot(iris,
                 kind='scatter',  # 散点图/回归分布图{'scatter', 'reg'})
                 diag_kind='hist',  # 直方图/密度图{'hist'， 'kde'}
                 hue='species',  # 按照某一字段进行分类
                 palette='husl',  # 设置调色板
                 markers=['o', 's', 'D'],  # 设置不同系列的点样式（这里根据参考分类个数）
                 size=2  # 图标大小
                 )
    # 只提取局部变量。
    g = sns.pairplot(iris, vars=['sepal_width', 'sepal_length'],
                     kind='reg', diag_kind='kde',
                     hue='species', palette='husl')
    # 其它参数设置
    sns.pairplot(iris, diag_kind='kde', markers='+',
                 plot_kws=dict(s=50, edgecolor='b', linewidth=1),
                 # 设置点样式
                 diag_kws=dict(shade=True)
                 )  # 设置密度图样式


# 4：可拆分绘制的散点图
def sns_demo4():
    # map_diag() + map_offdiag()
    g = sns.PairGrid(iris, hue='species', palette='hls',
                     vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    # 可筛选创建一个绘图表格区域，设置好x，y对应的数据，按照species分类

    # 对角线图表，plt.hist/sns.kdeplot
    g.map_diag(plt.hist,
               histtype='step',  # 可选：'bar','barstacked', 'step', 'stepfilled'
               linewidth=1, edgecolor='w')

    # 其它图表：plt.scatter/plt.bar...
    g.map_offdiag(plt.scatter, edgecolor='w', s=40, linewidth=1)
    # 设置点颜色、大小、描边宽度
    g.add_legend()  # 添加图例()


# 上三角和下三角#map_diag() + map_lower() + map_upper()
def sns_demo5():
    g = sns.PairGrid(iris)
    g.map_diag(sns.kdeplot, lw=3)  # 设置对角线图表
    g.map_upper(plt.scatter, color='r')  # 设置对角线上端图表
    g.map_lower(sns.kdeplot, cmap='Blues_d')  # 设置对角线下端图表


def miss_value():
    # pip install missingno
    import missingno as msno
    import pandas as pd
    import numpy as ny

    data = pd.read_csv("model.csv")
    # 无效矩阵的数据密集显示
    msno.matrix(data, labels=True, inline=False, sort='descending')
    # 条形图
    msno.bar(data)
    # 热图相关性 一个变量的存在或不存在如何强烈影响的另一个的存在
    # 关性为1，说明X5只要发生了缺失，那么X1.1也会缺失。 相关性为-1，说明X7缺失的值，那么X8没有缺失；而X7没有缺失时，X8为缺失。
    msno.heatmap(data)
    # 树状图 层次聚类算法通过它们的无效性相关性（根据二进制距离测量）将变量彼此相加，
    # 哪个组合最小化剩余簇的距离来分割变量。变量集越单调，它们的总距离越接近零，并且它们的平均距离（y轴）越接近零。
    msno.dendrogram(data)


def geo_graph():
    import folium
    oneUserMap = folium.Map(location=[40.07645623466996, 116.27861671489337], zoom_start=12)
    # 等值线图
    oneUserMap.choropleth(geo_path="geo_json_shape2.json",
                          data_out="data.json",
                          data=dty,
                          columns=["constituency", "count"],
                          key_on="feature.properties.PCON13NM.geometry.type.Polygon",
                          fill_color='PuRd',
                          fill_opacity=0.7,
                          line_opacity=0.2,
                          reset="True")


# pandas 数据状态参数预览
def pandas_dataviews():
    import pandas_profiling

    data = pd.read_csv("model.csv")
    profile = pandas_profiling.ProfileReport(data)
    profile.to_file(outputfile="output_file.html")


if __name__ == '__main__':
    geo_graph()
    exit(0)
    df = ts.get_hist_data('600848')
    # 1. 画图测试
    plot_line_scatter_demo(df["open"], df["high"])
    # 2. 蜡烛图测试
    pandas_candlestick_ohlc(df)
    # ~)c. 图示示例
    # # 1表. 用户，特征，分值
    # # 2表. 用户，特征，历史时间
    # ~)c. 图示示例 h1. 静态属性回归
    # # 1. score排序，索引重置
    # plt.scatter(range(train_df.shape[0]), np.sort(train_df[target_col].values))
    # # 2. score区间数量统计，分布曲线
    # sns.distplot(train_df[target_col].values, bins=50, kde=False, color="red")
    # # 3. 某特征不同离散值，对score的方差分布
    # sns.violinplot(x="feature_3", y=target_col, data=train_df)
    #
    # ~)c. 图示示例 h1. 历史属性回归
    # # 1. 时段，数量统计图(userid聚类)
    # cnt_srs = train_df['first_active_month'].dt.date.value_counts()
    # cnt_srs = cnt_srs.sort_index()
    # sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
    #
    # ~)c. 图示示例 h1. 静态join历史的聚合
    # # 1. userid历史量计数聚合 --> 相同数量的分值聚合求均值：数量 分值图
    # import plotly.offline as py
    # py.init_notebook_mode(connected=True)
    # import plotly.graph_objs as go
    #
    # gdf = hist_df.groupby("card_id")
    # gdf = gdf["purchase_amount"].size().reset_index()
    # # -->
    # cnt_srs = train_df.groupby("num_hist_transactions")[target_col].mean()
    # cnt_srs = cnt_srs.sort_index()
    #
    # # 2. 数量 分值图的区间方差图
    # bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 10000]
    # train_df['binned_num_hist_transactions'] = pd.cut(train_df['num_hist_transactions'], bins)
    #
    # sns.boxplot(x="binned_num_hist_transactions", y=target_col, data=train_df, showfliers=False)
    #

    # g = sns.JointGrid(x="binned_num_hist_transactions", y=target_col, data=train_df, ylim=gdpr)
    #
