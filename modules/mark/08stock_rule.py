# -*- coding: utf-8 -*-
from modules.stocks.finance_frame import navigation
from modules.stocks.stock_network import deep_network
from modules.stocks.stock_chara import gene_allpd
from modules.stocks.stock_learn import StockLearn
from txt.basic_mlp import npd_similar, nplot_timesq
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt


# import talib

# n = os.system(test.sh)
# n = os.system("dir")
# print(n)


def tmp_test():
    df = ts.get_hist_data('600848', start='2015-01-01', end='2015-12-31')
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    # 收市股价
    close = df.close
    # 每天的股价变动百分率
    ret = df.p_change / 100
    # 10日的移动均线为目标
    df['SMA_10'] = talib.MA(np.array(close), timeperiod=10)
    close10 = df.SMA_10
    # 处理信号
    SmaSignal = pd.Series(0, index=close.index)

    for i in range(10, len(close)):
        if all([close[i] > close10[i], close[i - 1] < close10[i - 1]]):
            SmaSignal[i] = 1
        elif all([close[i] < close10[i], close[i - 1] > close10[i - 1]]):
            SmaSignal[i] = -1

    SmaTrade = SmaSignal.shift(1).dropna()

    SmaBuy = SmaTrade[SmaTrade == 1]

    SmaSell = SmaTrade[SmaTrade == -1]

    SmaRet = ret * SmaTrade.dropna()

    # 累积收益表现
    # 股票累积收益率
    cumStock = np.cumprod(1 + ret[SmaRet.index[0]:]) - 1
    # 策略累积收益率
    cumTrade = np.cumprod(1 + SmaRet) - 1
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(cumStock, label="cumStock", color='k')
    plt.plot(cumTrade, label="cumTrade", color='r', linestyle=':')
    plt.title("股票累积收益率与10日平均策略收益率")
    plt.legend()
    exit(0)
    ts = pd.Series(np.random.randn(20) + 10, pd.date_range("2017-11-12", freq="D", periods=20))
    # 1. 序列处理，平移
    ts_lag = ts.shift()
    # 2. 均线
    ts_lag = ts.rolling(window=20)


def main():
    # 1. 生成特征
    parajson = {
        "avenlist": [5, 20],
        "labellist": [i for i in range(1, 129)],
    }
    label_pd = gene_allpd(parajson)
    # 2. 数据乱序
    splitn = 5
    datalenth = label_pd.shape[0]
    labelt_pd = label_pd[0:datalenth - int(datalenth / splitn)]
    labelv_pd = label_pd[datalenth - int(datalenth / splitn):]
    labelt_pd = shuffle(labelt_pd)
    # 2. 调用模型
    lclass = StockLearn()
    model_name = 'forest_base'
    pic_name = "valid_forest_base.png"
    lclass.forest_learn(labelt_pd, model_name)
    # 2.2 模型加载
    # print("predict train sets. length: %d" % labelt_pd.shape[0])
    # pdt_predict = lclass.forest_predict(labelt_pd, model_name)
    print("predict valid sets. length: %d" % labelv_pd.shape[0])
    pdv_predict = lclass.nforest_predict(labelv_pd, model_name)

    # 3. 评估
    # 3.1 不同 n个操作点， 预测 实际 上下限值 序列图。
    show_list = [i for i in pdv_predict.columns if i.startswith("predict_") or i.startswith("ylabel_")]
    nplot_timesq(pdv_predict[show_list])
    # 3.2 预测 实际 值 的相似分布。 均值方差描述
    # 3.3 不同 时间片 集合n 的偏离方差和均值
    # npd_similar(pdv_predict, "ttt123")
    # 4. 策略标准
    # 5. 自动交易
    # 6. 模型方式
    # 6.1 未来x分钟，涨跌y%.
    # 6.2 强化买卖点
    # 6.3 n只股票 各特征降维，
    # 6.4 未来走势


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    # import seaborn.regression as snsl
    from datetime import datetime
    import tushare as ts

    # 1. 数据选择
    # 1.1. 筛选集合
    stock_lis = ['300113', '300343', '300295', '300315']
    end = datetime.today()  # 开始时间结束时间，选取最近一年的数据
    start = datetime(end.year - 1, end.month, end.day)
    end = str(end)[0:10]
    start = str(start)[0:10]
    df = pd.DataFrame()
    for stock in stock_lis:
        closing_df = ts.get_hist_data(stock, start, end)['close']
        df = df.join(pd.DataFrame({stock: closing_df}), how='outer')
    tech_rets = df.pct_change()
    print(df.head(3))
    print(df.tail(3))
    print(tech_rets.head(3))
    print(tech_rets.tail(3))

    # pearson相关热图
    rets = tech_rets.dropna()
    plt.figure(1)
    sns.heatmap(rets.corr(), annot=True)
    plt.draw()
    # plt.close(1)
    # 收益风险图
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

    # snsl.corrplot(tech_rets.dropna())
    # 1.2. 时段聚类α，β
    # 1.2. 时段聚类α，β
    # 遗传因子选特征
    # 参数策略组合迭代回测
    # 精确拟合度

    # 2. 网络结构
    # 2.1. (原始+深户)输入16 *log； 便于卷积
    # 2.2. 长度为2的每维 卷积核valid step2，10-50个；历史收集 chara4层
    # 2.3. catch 输入+卷积各层，full+-lrelu；迭代 2次 基层策略 出100
    # 2.4. catch 3的relu各层 all dim，full drop 0.1-0.5 +-lrelu；迭代 2次 高层策略 出1000 出512

    # 3. 回测 交易次数 单次均值是方差 年化值 置信度 置信区间
    # 3.1 学习回归值 历史方差
    # 3.2 方向概率 准确度

    exit(0)

    main()
    # tmp_test()
    # deep_network()
    # panda_get_data()
    # startdate = "2018-02-10"
    # all_store(startdate)
    # tmp_test()
    # TSstockdata()
    # navigation()
    # exit(0)
    # store_recover()
    # for i in ["000001"]:
    #     single_store(i)
    #     # single_stock_type("600848", "D")
    #     # single_store(code, type)
