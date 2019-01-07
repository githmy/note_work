# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os

import numpy as np
import pandas as pd
import simplejson
import statsmodels.tsa.stattools as tsat
from modules.stocks.stock_data import Stockdata
from modules.stocks.stock_learn import StockLearn
from modules.stocks.stock_mlp import plot_timesq, pd_similar
from sklearn.utils import shuffle

# from get_data import scrap_all_store
# from modules.mystrategy import MyStrategy
# from pyalgotrade import broker
# from pyalgotrade import plotter

cmd_path = os.getcwd()
data_path = os.path.join(cmd_path, "..", "..", "..", "nocode", "customer")
data_opath = os.path.join(data_path, "input", "data")
data_path_recover = os.path.join(data_opath, "recover")
data_path_res = os.path.join(data_opath, "res")
data_path_stock = os.path.join(data_opath, "stock")
datalogfile = os.path.join(data_path, "logs", 'finance_analysis.log')

# 创建一个logger
logger1 = logging.getLogger('logger_out')
logger1.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(datalogfile)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger1.addFilter(filter)
logger1.addHandler(fh)
logger1.addHandler(ch)


def read_file(filename, encoding="utf-8-sig"):
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def read_json_file(filename):
    """Read json from a file."""
    content = read_file(filename)
    try:
        return simplejson.loads(content)
    except ValueError as e:
        raise ValueError("Failed to read json from '{}'. Error: "
                         "{}".format(os.path.abspath(filename), e))


def adfuller_test():
    """
    statsmodels.tsa.stattools.adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)[source]¶
     x: 序列，一维数组
     maxlag：差分次数
     regresion:{c:只有常量，
                ct:有常量项和趋势项，
                ctt:有常量项、线性和二次趋势项，
                nc:无任何选项}
     autolag:{aic or bic: default, then the number of lags is chosen to minimize the corresponding information criterium,
              None:use the maxlag,
              t-stat:based choice of maxlag. Starts with maxlag and drops a lag until the t-statistic on the last lag length is significant at the 95 % level.}
    :return: 
    """
    global data_path_stock
    tpath = data_path_stock
    coden = "000001"
    filePath = coden + "_" + "5" + ".csv"
    tmp_path = os.path.join(tpath, filePath)
    df = pd.read_csv(tmp_path, header=0, encoding="utf8")
    # print(df.head())
    aa = tsat.adfuller(df["open"], 1)
    print(aa)


def hurst(ts):
    "adfuller_test 的另一种方法 H>0.5 延续趋势，H<0.5 逆转趋势。"
    "Returns the Hurst Exponent of the time series vector ts"
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def strategy_among_instance_single_case(stype, numf, numt, bfamp, bfnum, timef, timestat, timeverify, col,
                                        stock_info_array, datalists):
    """
    样本间的 特征极值统计，如最小市值。
    :param stype: 功能类型
    :param numf: 排行起
    :param numt: 排行止
    :param bfamp: 采样点前置振幅
    :param bfnum: 采样点前置位置
    :param timef: 时间段起
    :param timestat: 统计步数
    :param timeverify: 测试步数
    :param col: 功能载体执行列
    :param datalists: 功能执行的载体列表 
    :return 0: 
    """
    # 1. 策略结果的特征值提取
    stock_info = stock_info_array[0]
    stock_info.set_index("code", inplace=True)
    # 1.1 特征值生成
    for i1 in datalists:
        # datalists[i1].insert(1, "liquid", 1.0)
        # datalists[i1].insert(0, "liquid", 1.3)
        datalists[i1][col] = stock_info.loc[i1, col] / datalists[i1]["close"][
            datalists[i1].shape[0] - 1] * datalists[i1]["close"]
    # 1.2 流动量提取
    tmp_obj = [datalists[i].rename(columns={col: i})[i] for i in datalists]
    liquids_pd = pd.concat(tmp_obj, axis=1)
    # 1.3 流动量转置
    liquids_pd = liquids_pd.T
    liquids_pd.to_csv(os.path.join(data_path_res, "liquids_order.csv"))
    # 1.4 流动量排序号
    orderl_pd = pd.DataFrame(data={})
    for i in liquids_pd:
        orderl_pd[i] = liquids_pd[i].rank(ascending=1, method='first')
        orderl_pd.reset_index(level=0, inplace=True)  # （the first）index 改为 column
        orderl_pd[[i]] = orderl_pd[[i]].fillna(1e6).astype(int)
        orderl_pd.set_index(i, inplace=True)
        orderl_pd.sort_index(axis=0, ascending=True, inplace=True)
        plotlist = [datalists[i2].rename(columns={"close": i2})[i2] for i2 in orderl_pd[numf:numt]["index"]]
        plotlist_pd = pd.concat(plotlist, axis=1)
        # 归一化
        for i2 in plotlist_pd.columns:
            plotlist_pd[i2] = plotlist_pd[i2] / plotlist_pd.loc[i, i2]
        lenthcolumn = len(plotlist_pd.columns)
        # 列求和
        plotlist_pd['ave' + str(lenthcolumn)] = plotlist_pd.apply(lambda x: x.sum() / lenthcolumn, axis=1)
        plotlist_pd['log(ave)' + str(lenthcolumn)] = np.log(plotlist_pd['ave' + str(lenthcolumn)])
        # # 行求和
        # plotlist_pd.loc['Row_sum'] = plotlist_pd.apply(lambda x: x.sum() / len(liquids_pd.columns))
        # print(plotlist_pd)
        # 取一天的前n，画时间图。
        plot_timesq(
            pd.concat([plotlist_pd['ave' + str(lenthcolumn)], plotlist_pd['log(ave)' + str(lenthcolumn)]], axis=1))
        break


def strategy_among_instance(stype, numf, numt, col, stock_info_array, datalists):
    """
    样本间的 特征极值统计，如最小市值。
    :param stype: 功能类型
    :param num: 功能参数
    :param col: 功能载体执行列
    :param datalists: 功能执行的载体列表 
    :return 0: 
    """
    # 1. 策略结果的特征值提取
    stock_info = stock_info_array[0]
    stock_info.set_index("code", inplace=True)
    # 1.1 特征值生成
    for i1 in datalists:
        # datalists[i1].insert(1, "liquid", 1.0)
        # datalists[i1].insert(0, "liquid", 1.3)
        datalists[i1][col] = stock_info.loc[i1, col] / datalists[i1]["close"][
            datalists[i1].shape[0] - 1] * datalists[i1]["close"]
    # 1.2 流动量提取
    tmp_obj = [datalists[i].rename(columns={col: i})[i] for i in datalists]
    liquids_pd = pd.concat(tmp_obj, axis=1)
    # 1.3 流动量转置
    liquids_pd = liquids_pd.T
    print(liquids_pd.head())
    tmpq_path = os.path.join(data_path_res, "liquids_mount.csv")
    if os.path.isfile(tmpq_path):
        os.remove(tmpq_path)
    liquids_pd.to_csv(tmpq_path)
    tmpo_path = os.path.join(data_path_res, "liquids_order.csv")
    if os.path.isfile(tmpo_path):
        os.remove(tmpo_path)
    liquids_pd.to_csv(tmpo_path)
    # 1.4 流动量排序号
    orderl_pd = pd.DataFrame(data={})
    for i in liquids_pd:
        orderl_pd[i] = liquids_pd[i].rank(ascending=1, method='first')
        orderl_pd.reset_index(level=0, inplace=True)  # （the first）index 改为 column
        orderl_pd[[i]] = orderl_pd[[i]].fillna(1e6).astype(int)
        orderl_pd.set_index(i, inplace=True)
        orderl_pd.sort_index(axis=0, ascending=True, inplace=True)
        plotlist = [datalists[i2].rename(columns={"close": i2})[i2] for i2 in orderl_pd[numf:numt]["index"]]
        plotlist_pd = pd.concat(plotlist, axis=1)
        # 归一化
        for i2 in plotlist_pd.columns:
            plotlist_pd[i2] = plotlist_pd[i2] / plotlist_pd.loc[i, i2]
        lenthcolumn = len(plotlist_pd.columns)
        # 列求和
        plotlist_pd['ave' + str(lenthcolumn)] = plotlist_pd.apply(lambda x: x.sum() / lenthcolumn, axis=1)
        plotlist_pd['log(ave)' + str(lenthcolumn)] = np.log(plotlist_pd['ave' + str(lenthcolumn)])
        # # 行求和
        # plotlist_pd.loc['Row_sum'] = plotlist_pd.apply(lambda x: x.sum() / len(liquids_pd.columns))
        # print(plotlist_pd)
        # 取一天的前n，画时间图。
        plot_timesq(
            pd.concat([plotlist_pd['ave' + str(lenthcolumn)], plotlist_pd['log(ave)' + str(lenthcolumn)]], axis=1))
        break
    # print(orderl_pd)
    # # # 3. 检验
    # # 取一天的前n小，画时间图。不同的未来跳跃取特征
    # orderl_pd["2015-03-02"]
    # orderl_pd.reset_index(level=0, inplace=True)  # （the first）index 改为 column
    # orderl_pd.set_index("order", inplace=True)
    # orderl_pd.sort_index(axis=0, ascending=True, inplace=True)
    # plotlist = [datalists[i] for i in orderl_pd[0:num, "2015-03-02"]]
    # plotlist_pd = pd.concat(plotlist, axis=1)
    # print(plotlist_pd.head())
    # plotlist_pd = plotlist_pd["2015-03-02":, :]
    # # 取一天的前n，画时间图。
    # print(plotlist_pd.head())
    # plot_timesq(plotlist_pd)
    # # x = datalists["000001"]["close"]
    # # print(x)
    # # y = np.diff(x, n=1)
    # # print(y)
    # # ts_log = np.log(x)
    # # ts_diff = ts_log.diff(1)
    # # ts_diff.dropna(inplace=True)
    # # print(ts_diff)

    # typedict = {"date": np.datetime_data,"liquids": np.float64}
    # figdata = pd.read_csv(tmpfile, header=0, encoding="utf8", parse_dates=[0], index_col=0, dtype=typedict)
    # for i1 in datalists:
    #     print(i1)
    #     print(datalists[i1])
    #     break
    # 2. sklearn回归，得出 概率，期望方差重尾峰值
    # 3. 结果的金凯利公式组合，得出收益的期望和方差概率
    return 0


# def runStrategy():
#     # 1.下载数据
#     # 2.获得回测数据，
#     feed = yahoofeed.Feed()
#     feed.addBarsFromCSV("jdf", "jdf.csv")
#     # 3.broker setting
#     # 3.1 commission类设置
#     # a.没有手续费
#     # broker_commission = pyalgotrade.broker.backtesting.NoCommission()
#     # b.amount：每笔交易的手续费
#     # broker_commission = pyalgotrade.broker.backtesting.FixedPerTrade(amount)
#     # c.百分比手续费
#     broker_commission = broker.backtesting.TradePercentage(0.0001)
#     # 3.2 fill strategy设置
#     fill_stra = broker.fillstrategy.DefaultStrategy(volumeLimit=0.1)
#     sli_stra = broker.slippage.NoSlippage()
#     fill_stra.setSlippageModel(sli_stra)
#     # 3.3完善broker类
#     brk = broker.backtesting.Broker(1000000, feed, broker_commission)
#     brk.setFillStrategy(fill_stra)
#
#     # 4.把策略跑起来
#     myStrategy = MyStrategy(feed, "jdf", brk)
#     # Attach a returns analyzers to the strategy.
#     trade_situation = trades.Trades()
#     myStrategy.attachAnalyzer(trade_situation)
#
#     returnsAnalyzer = returns.Returns()
#     myStrategy.attachAnalyzer(returnsAnalyzer)
#
#     # Attach the plotter to the strategy.
#     plt = plotter.StrategyPlotter(myStrategy)
#     plt.getInstrumentSubplot("jdf")
#     plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
#
#     myStrategy.run()
#     myStrategy.info("Final portfolio value: $%.2f" % myStrategy.getResult())
#     plt.plot()


def navigation():
    """
    决策总框架, 每一个子策略，1.提取特征，2.数据清洗，3.学习返回 操作指标.收益概率.期望.方差，4.子策略组合
    :return 0: 
    """

    # 1.读配置文件
    fileenv = os.path.join(os.getcwd(), "08config.json")
    file_config = read_json_file(fileenv)
    print(file_config)

    # 0. 模型路由
    # modeltype = "nn"
    # modeltype = "hmm"
    # modeltype = "forest"
    # modeltype = "keras"
    modeltype = "iter"

    # 1. 数据加载
    dclass = Stockdata()
    stock_info = dclass.data_stock_info()
    # print(stock_info.columns)
    print(stock_info.head(1))
    # 1.1. 不同数据获取
    stocklist = dclass.data_stocklist()[0:1]
    print(stocklist)
    # stocklist = dclass.data_stocklist()
    data_list = dclass.data_stocklist_value("D", stocklist)
    scode = "000001"
    # stpye = "D"
    # tmpfile = os.path.join(data_path_res, scode + "_" + stpye + "feature.csv")
    # label_pd = dclass.data_with_labels(scode)
    tmppdlist = []
    if modeltype == "hmm":
        for [i1] in stocklist:
            tmppdlist.append(dclass.data_without_labels(i1))
    elif modeltype == "nn":
        for [i1] in stocklist:
            tmppdlist.append(dclass.data_with_labels(i1))
    elif modeltype == "forest":
        for [i1] in stocklist:
            tmppdlist.append(dclass.data_with_labels(i1))
    elif modeltype == "keras":
        for [i1] in stocklist:
            tmppdlist.append(dclass.data_with_labels(i1))
    elif modeltype == "iter":
        # 1. 名单都按各指标排序，前几的规律。
        # 最小：市盈率，市盈率动态，流通市值，市净率
        # 2. 预期值 现值，N日变幅 的拟合损失
        # 3. 规律板块聚类。
        # 3.1 单支一阶微分，规律还原的N日拟合损失
        for [i1] in stocklist:
            tmppdlist.append(dclass.data_with_labels(i1))
    else:
        pass
    label_pd = pd.concat(tmppdlist, axis=0)
    print(label_pd.columns)
    # 1.2. 训练测试拆分
    splitn = 5
    splits = "2015-02-01"
    # splits = None
    splitb = "2017-02-01"
    splita = "2017-02-02"
    print("date from %s to %s ,valid from % s" % (splits, splitb, splita))
    if splita is None:
        datalenth = len(label_pd["p_change"])
        labelt_pd = label_pd[0:datalenth - int(datalenth / splitn)]
        labelv_pd = label_pd[datalenth - int(datalenth / splitn):]
    else:
        labelt_pd = label_pd[splits:splitb]
        labelv_pd = label_pd[splita:]
    if modeltype == "hmm":
        pass
    elif modeltype == "nn":
        pass
    elif modeltype == "forest":
        labelt_pd = shuffle(labelt_pd)
    elif modeltype == "keras":
        labelt_pd = shuffle(labelt_pd)
    elif modeltype == "iter":
        labelt_pd = shuffle(labelt_pd)
    else:
        pass

    # 2. 模型核心
    lclass = StockLearn()
    if modeltype == "hmm":
        # # 2.1 学习模型
        model_name = 'hmm_base'
        pic_name = "valid_hmm_base.png"
        lclass.hmm_learn(labelt_pd, model_name)
        print("predict valid sets. length: %d" % labelv_pd.shape[0])
        pdv_predict = lclass.hmm_predict(labelv_pd, model_name)
    elif modeltype == "nn":
        # # 2.1 学习模型
        model_name = 'nn_base'
        pic_name = "valid_nn_base.png"
        lclass.nn_learn(labelt_pd, model_name)
        print("predict valid sets. length: %d" % labelv_pd.shape[0])
        pdv_predict = lclass.nn_predict(labelv_pd, model_name)
    elif modeltype == "forest":
        labelt_pd = shuffle(labelt_pd)
        model_name = 'forest_base'
        pic_name = "valid_forest_base.png"
        lclass.forest_learn(labelt_pd, model_name)
        # 2.2 模型加载
        # print("predict train sets. length: %d" % labelt_pd.shape[0])
        # pdt_predict = lclass.forest_predict(labelt_pd, model_name)
        print("predict valid sets. length: %d" % labelv_pd.shape[0])
        pdv_predict = lclass.forest_predict(labelv_pd, model_name)
    elif modeltype == "keras":
        # # 2.1 学习模型
        model_name = 'nn_base'
        pic_name = "valid_nn_base.png"
        lclass.nn_learn(labelt_pd, model_name)
        print("predict valid sets. length: %d" % labelv_pd.shape[0])
        pdv_predict = lclass.nn_predict(labelv_pd, model_name)
    elif modeltype == "iter":
        # # 2.1 学习模型
        model_name = 'nn_base'
        pic_name = "valid_nn_base.png"
        lclass.nn_learn(labelt_pd, model_name)
        print("predict valid sets. length: %d" % labelv_pd.shape[0])
        pdv_predict = lclass.nn_predict(labelv_pd, model_name)
    else:
        pass

    pred_pd = pd.concat([labelt_pd, pdv_predict], axis=0)
    # 曲线拼接
    for i1 in pred_pd.columns:
        if i1.startswith("ylabel_"):
            pred_pd["predict_" + i1][splits:splitb] = pred_pd[i1[7:]][splits:splitb]
            pred_pd[i1][splits:splitb] = pred_pd[i1[7:]][splits:splitb]
    # 3 画图
    plot_timesq(pred_pd[["predict_ylabel_p_change", "ylabel_p_change"]])
    plot_timesq(pred_pd[["predict_ylabel_close", "ylabel_close"]])
    pd_similar(pdv_predict, pic_name)

    # # plot画图
    # # y = np.array(Yv["ylabel_p_change"]).reshape(-1)
    # y = pdv_predict["ylabel_p_change"]
    # yv_hat = pdv_predict["predict_ylabel_p_change"]
    # plot_similar(y, yv_hat, vfile)
    # # expection = self.check_expection(y, y_hat)
    # # print(expection)

    # 4 其他方法
    # feature_pd = dclass.data_feature(scode)
    # xd, yd = feature_pd[scode + "_Dfeature"].shape
    # 1. 策略尝试
    # runStrategy()
    # clf = RandomForestRegressor(n_estimators=200, criterion='entropy', max_depth=3)
    # clf.fit(np.array(Xt), np.array(Yt["ylabel_p_change"]))
    # print('\t预测正确数目：', c)
    # print('\t准确率: %.2f%%' % (100 * float(c) / float(len(y))))
    exit(0)

    # 2. 检测效果
    middle_pd = dclass.data_middle()
    trade_frame(data_list, middle_pd, feature_pd)
    # 1.1 神经网络 另一个文件python3的版本tensorflow
    # 1.2 极值测试
    stype = "extreme"
    numf = 0
    numt = 5
    col = "liquidAssets"
    strategy_among_instance(stype, numf, numt, col, [stock_info], data_list)
    # strategy_among_instance_single_case(stype, numf, numt, bfamp, bfnum, timef, timestat, timeverify, col,[stock_info], data_list)
    # 1.2 周期振幅分析
    # 1.3 策略特质利用选取
    # 2. 策略特质利用选取
    # 3. 策略组合
    return 0


# 给定时间线
# def trade_frame(timeline, weigh, start, end, stratges):
def trade_frame(data_list, middle_pd, feature_pd):
    """
    根据策略查看结果
    :param data_list:pd的时间序列[所有]
    :param middle_pd:初级加工数据[data_list]
    :param weigh:{选定的组合：的权重}
    :param start: 起始索引
    :param end: 结束索引
    :param stratges: 策略组
    :return:倍数 
    """
    # 决策树不可能，因为只是部分[某种策略，]
    profit = 1
    nc = 0.02
    gc = 0.002
    stratges = {
        "maintain": 0,
        "linein": 0,
        "login": 0,
        "logout": 0,
        "lineout": 0,
    }
    # 1. 数据区段截取
    # data_list
    # middle_pd
    # timeline = timeline.ix[start:]
    # 2. 策略生成数据
    # 2.1 简单策略
    print(feature_pd)
    tmp_obj = []
    for i in range(0, len(timeline["close"]) - 1):
        if stratges["linein"] > (1 + nc * i) and stratges["login"] > gc * i:
            pass
        elif stratges["lineout"] > (1 + nc * i) and stratges["logout"] > gc * i:
            pass
        tmp_obj.append(0)

    # 3. 策略数据写入
    profit_pd = pd.concat(tmp_obj, axis=1)
    tmps_path = os.path.join(data_path_res, "stratge_profit.csv")
    if os.path.isfile(tmps_path):
        os.remove(tmps_path)
    profit_pd.to_csv(tmps_path, index=True, index_label="code")
    print("file: %s" % tmps_path)

    return profit_pd


def prepare_data():
    # 1. 原始数据抓取
    startdate = '2018-02-05'
    exit(0)
    # 2. 中间数据生成
    dclass = Stockdata()
    # scode = "000001"
    # stpye = "D"
    # tmpfile = os.path.join(data_path_res, scode + "_" + stpye + "feature.csv")
    dclass.generate_feature("000001")
    # dclass.generate_middles()
    # scrap_all_store(startdate)
    return 0


if __name__ == '__main__':
    # # 1. 随机游走测试
    # adfuller_test()
    # # 3. 预处理数据
    # prepare_data()
    # 4. 模型主框架
    navigation()
