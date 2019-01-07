from modules.stocks.finance_frame import navigation
from modules.stocks.stock_network import deep_network
from modules.stocks.stock_data import TSstockScrap
from modules.stocks.stock_chara import gene_allpd
from modules.stocks.stock_learn import StockLearn
from modules.stocks.stock_paras import parseArgs, bcolors, get_paras
from modules.stocks.stock_mlp import npd_similar, nplot_timesq
import tushare as ts
from sklearn.utils import shuffle
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Finan_frame(object):
    """
    学习管道的实现：
    1. 精细特征，机器学习
    2. 特征加工，早期版深度学习
    3. log输入，深度学习
    4. 原始输入，强化学习
    """
    map_purpose = {
        "getdata": None,
        "learning": None,
        "test": None,
        "trade": None,
    }
    map_data = {
        "mla": None,
        "edl": None,
        "dl": None,
        "rf": None,
    }
    map_chara = {
        "mla": None,
        "edl": None,
        "dl": None,
        "rf": None,
    }
    map_learn = {
        "mla": None,
        "edl": None,
        "dl": None,
        "rf": None,
    }

    def __init__(self, parajson):
        # 1. 参数解析
        self.__parameters = parajson
        # 2. 数据下载，爬取，读取
        self.__scrap_data(self.__parameters)
        # 4. 筛选数据
        self.__data_filter(self.__parameters)
        # 5. 特征生成
        self.__get_chara(self.__parameters)
        # 6. 学习规律
        self.__get_learn(self.__parameters)
        # 7. 回测
        self.__back_test(self.__parameters)
        # 8. 自动交易
        self.__trade_fun(self.__parameters)

    def __scrap_data(self, para):
        if para["scrap_data"]["usesig"] == 0:
            return 0
        print()
        print("*" * 60)
        print("begin __scrap_data")
        scrap_data_class = TSstockScrap(para["process"]["nocode_path"])
        if para["scrap_data"]["way"]["normal"] == 1:
            scrap_data_class.scrap_all_n_store(para["scrap_data"]["start_date"])
        elif para["scrap_data"]["way"]["hist"] == 1:
            scrap_data_class.scrap_all_h_store(para["scrap_data"]["start_date"])
        elif para["scrap_data"]["way"]["web"] == 1:
            pass
        else:
            pass
        print("finished __scrap_data")
        print("*" * 60)

    def __data_filter(self, para):
        if para["data_filter"]["usesig"] == 0:
            return 0
        print("in __data_filter")
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

    def __get_chara(self, para):
        if para["get_chara"]["usesig"] == 0:
            return 0
        print("in __get_chara")
        # 1.2. 时段聚类α，β
        # 遗传因子选特征

    def __get_learn(self, para):
        if para["get_learn"]["usesig"] == 0:
            return 0
        print("in __get_learn")
        # 2. 网络结构
        # 2.1. (原始+深户)输入16 *log； 便于卷积
        # 2.2. 长度为2的每维 卷积核valid step2，10-50个；历史收集 chara4层
        # 2.3. catch 输入+卷积各层，full+-lrelu；迭代 2次 基层策略 出100
        # 2.4. catch 3的relu各层 all dim，full drop 0.1-0.5 +-lrelu；迭代 2次 高层策略 出1000 出512

    def __back_test(self, para):
        if para["back_test"]["usesig"] == 0:
            return 0
        print("in __back_test")
        # 参数策略组合迭代回测
        # 精确拟合度
        # 3. 回测 交易次数 单次均值是方差 年化值 置信度 置信区间
        # 3.1 学习回归值 历史方差
        # 3.2 方向概率 准确度

    def __trade_fun(self, para):
        if para["trade_fun"]["usesig"] == 0:
            return 0
        print("in __trade_fun")


def main(args=None):
    # 1. 命令行
    parajson = get_paras(args)
    print(parajson)
    # 2. 流程解析类
    print("all start".center(20, " ").center(200, "#"))
    finish = Finan_frame(parajson)
    print("all end".center(20, " ").center(200, "#"))
    return 0

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


if __name__ == '__main__':
    # 1. 参数解析
    main(sys.argv[1:])
    # deep_network()
    # navigation()
