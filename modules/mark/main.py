from modules.stocks.finance_frame import navigation
from modules.stocks.stock_network import deep_network
from modules.stocks.stock_data import TSstockScrap, LocalStockdata
from modules.stocks.stock_chara import Component_charas
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

    def __init__(self, parajson):
        # 1. 参数解析
        self.__parameters = parajson
        self.__scrap_data_class = TSstockScrap(self.__parameters["process"]["nocode_path"])
        # 2. 数据下载，爬取，读取
        self.__scrap_data(self.__parameters)
        # 4. 筛选数据
        self.__stock_list = None
        self.__ori_datas = None
        self.__start = None
        self.__valid = None
        self.__test = None
        self.__end = None
        self.__local_data_class = LocalStockdata(self.__parameters["process"]["nocode_path"])
        self.__data_filter(self.__parameters)
        # 5. 特征生成
        self.__chara_class = Component_charas()
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
        # 获取网络数据
        # scrap_data_class = TSstockScrap(para["process"]["nocode_path"])
        if para["scrap_data"]["way"]["normal"] == 1:
            self.__scrap_data_class.scrap_all_n_store(para["scrap_data"]["start_date"])
        elif para["scrap_data"]["way"]["hist"] == 1:
            self.__scrap_data_class.scrap_all_h_store(para["scrap_data"]["start_date"])
        elif para["scrap_data"]["way"]["web"] == 1:
            raise Exception("not yet!")
        else:
            pass
        print("finished __scrap_data")
        print("*" * 60)

    def __data_filter(self, para):
        if para["data_filter"]["usesig"] == 0:
            return 0
        print()
        print("*" * 60)
        print("begin __data_filter")
        # 1. 数据选择
        self.__stock_list = list(np.squeeze(self.__local_data_class.data_stocklist()))
        # 1.1. 过滤st
        if para["data_filter"]["way"]["st"] == 1:
            st_stocklist = list(np.squeeze(self.__local_data_class.data_ST_list()))
            self.__stock_list = [i1 for i1 in self.__stock_list if i1 not in st_stocklist]
        if para["data_filter"]["way"]["lastopen"] == 1:
            # stop_list = self.__scrap_data_class.stop_stock()
            stop_list = list(np.squeeze(self.__scrap_data_class.stop_stock()))
            print("stop_list:", stop_list)
            self.__stock_list = [i1 for i1 in self.__stock_list if i1 not in stop_list]
            print("all_list:", self.__stock_list)
        print("finished __data_filter")
        print("*" * 60)

    def __get_chara(self, para):
        if para["get_chara"]["usesig"] == 0:
            return 0
        print()
        print("*" * 60)
        print("begin __get_chara")
        # 1.1. 数据读入
        self.__ori_datas = self.__local_data_class.data_stocklist_value("D", self.__stock_list)
        # 1.2. 特征获取
        self.__data_and_char(para)
        # 1.3. 数据集切分
        self.__data_split(para)
        # 1.4. 时段聚类α，β
        self.__data_cluster(para)
        # 遗传因子选特征
        print("finished __get_chara")
        print("*" * 60)

    def __data_and_char(self, para):
        df = pd.DataFrame()
        if para["get_chara"]["way"]["mla"] == 1:
            self.__ori_datas = self.__chara_class.mla_charas(self.__ori_datas,
                                                             para["get_chara"]["way"]["charparas"])
            return 0
        if para["get_chara"]["way"]["dl"] == 1:
            self.__ori_datas = self.__chara_class.deeplearn_charas(self.__ori_datas,
                                                                   para["get_chara"]["way"]["charparas"])
            return 0
        if para["get_chara"]["way"]["rf"] == 1:
            self.__ori_datas = self.__chara_class.reforce_charas(self.__ori_datas,
                                                                 para["get_chara"]["way"]["charparas"])
            return 0

    def __data_split(self, para):
        if para["get_chara"]["date"]["start"] is None:
            pass
        if para["get_chara"]["date"]["valid"] is None:
            return 0
        if para["get_chara"]["date"]["test"] is None:
            return 0
        if para["get_chara"]["date"]["end"] is None:
            pass
        self.__train_datas = self.__local_data_class.data_stocklist_value(stpye, nplist)
        self.__valid_datas = self.__local_data_class.data_stocklist_value(stpye, nplist)
        self.__test_datas = self.__local_data_class.data_stocklist_value(stpye, nplist)
        end = datetime.today()  # 开始时间结束时间，选取最近一年的数据
        start = datetime(end.year - 1, end.month, end.day)
        end = str(end)[0:10]
        start = str(start)[0:10]

    def __data_cluster(self, para):
        # todo: 调用相似内积函数求矩阵
        if para["get_chara"]["cluster"]["use"] == 0:
            return 0
            # 调用相似内积函数求矩阵

    def __get_learn(self, para):
        if para["get_learn"]["usesig"] == 0:
            return 0
        print()
        print("*" * 60)
        print("begin __get_learn")
        pass
        # 2. 网络结构
        # 2.1. (原始+深户)输入16 *log； 便于卷积
        # 2.2. 长度为2的每维 卷积核valid step2，10-50个；历史收集 chara4层
        # 2.3. catch 输入+卷积各层，full+-lrelu；迭代 2次 基层策略 出100
        # 2.4. catch 3的relu各层 all dim，full drop 0.1-0.5 +-lrelu；迭代 2次 高层策略 出1000 出512
        print("finished __get_learn")
        print("*" * 60)

    def __back_test(self, para):
        if para["back_test"]["usesig"] == 0:
            return 0
        print()
        print("*" * 60)
        print("begin __back_test")
        pass
        # 参数策略组合迭代回测
        # 精确拟合度
        # 3. 回测 交易次数 单次均值是方差 年化值 置信度 置信区间
        # 3.1 学习回归值 历史方差
        # 3.2 方向概率 准确度
        print("finished __back_test")
        print("*" * 60)

    def __trade_fun(self, para):
        if para["trade_fun"]["usesig"] == 0:
            return 0
        print()
        print("*" * 60)
        print("begin __trade_fun")
        pass
        print("finished __trade_fun")
        print("*" * 60)


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
