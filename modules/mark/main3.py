import sys, os
import datetime
import json
from modules.portfolio import Portfolio
from modules.event import *
from modules.datahandle import CSVDataHandler, CSVAppendDataHandler, LoadCSVHandler
from modules.strategys import MovingAverageCrossStrategy, MultiCrossStrategy, MlaStrategy
from modules.executions import SimulatedExecutionHandler
from modules.backtests import Backtest, LoadBacktest
from utils.log_tool import *
import tushare as ts


def choice_list():
    # 行业分类
    tt = ts.get_industry_classified()
    infos = tt['code'].groupby([tt['c_name']]).apply(list)
    industjson = json.loads(infos.to_json(orient='index', force_ascii=False))
    # 中小板分类
    tt = ts.get_sme_classified()
    smartlist = list(set(tt["code"]).intersection(set(industjson["电子信息"])))
    # # 概念分类
    # tt = ts.get_concept_classified()
    # infos = tt['code'].groupby([tt['c_name']]).apply(to_list)
    # conceptjson = json.loads(infos.to_json(orient='index', force_ascii=False))
    # # 小轻，突破边缘的信息类
    # nearbreaklist = list(set(smartlist).intersection(set(conceptjson["智能机器"])))
    # print(conceptjson["智能机器"])
    # print(nearbreaklist)
    return smartlist


class Acount(object):
    def __init__(self, config):
        self.account = config["account"]
        self.func_type = config["data_ori"]["func_type"]
        self.test_type = config["back_test"]["test_type"]
        self.start_train = config["back_test"]["start_train"]
        self.end_train = config["back_test"]["end_train"]
        self.start_predict = config["back_test"]["start_predict"]
        self.end_predict = config["back_test"]["end_predict"]
        self.initial_capital = config["back_test"]["initial_capital"]
        self.heartbeat = config["back_test"]["heartbeat"]
        self.data_type = config["data_ori"]["data_type"]
        self.csv_dir = config["data_ori"]["csv_dir"]
        self.symbol_list = config["data_ori"]["symbol_list"]
        self.ave_list = config["data_ori"]["ave_list"]
        self.bband_list = config["data_ori"]["bband_list"]
        self.ret_list = config["data_ori"]["ret_list"]
        self.stratgey_name = config["stratgey"]["stratgey_name"]
        self.portfolio_name = config["portfolio"]["portfolio_name"]
        # 生成标准参数
        self._gene_stand_paras()

    def _gene_stand_paras(self):
        pass

    def _get_train_list(self):
        flist = []
        for root, dirs, files in os.walk(data_path, topdown=True):
            flist = [i1.replace(".csv", "") for i1 in files if i1.endswith("_D.csv")]
            break
        # flist = flist[0:len(flist) // 2]
        return flist

    def __call__(self, *args, **kwargs):
        # 1. 判断加载模型
        backtest = None
        if self.test_type == "实盘":
            pass
        elif self.test_type == "模拟":  # 已有数据模式
            if self.data_type == "实盘demo":  # 已有数据，动态模拟, 原始例子
                backtest = Backtest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.ret_list,
                    CSVDataHandler, SimulatedExecutionHandler, Portfolio, MovingAverageCrossStrategy)
            elif self.data_type == "实盘":  # 已有数据，动态模拟, 未完善
                backtest = Backtest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.ret_list,
                    CSVAppendDataHandler, SimulatedExecutionHandler, Portfolio, MultiCrossStrategy)
            elif self.data_type == "模拟":  # 已有数据，直观统计
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    # self.csv_dir, self.symbol_list, self.ave_list, self.bband_list,
                    self.csv_dir, self._get_train_list(), self.ave_list, self.bband_list,
                    # self.csv_dir, choice_list(), self.ave_list, self.bband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy)
            elif self.data_type == "网络":  # 已有数据，统计强化学习
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    None, self._get_train_list(), self.ave_list, self.bband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy)
            else:
                raise Exception("error data_type 只允许：实盘demo, 实盘, 模拟, 网络")
        else:
            raise Exception("error test type.")
        # 2. 判断执行功能
        if self.func_type == "train":
            backtest.train()
        elif self.func_type == "backtest":
            backtest.simulate_trading()
        elif self.func_type == "lastday":
            para_config = {
                "hand_unit": 100,
                "initial_capital": 10000.0,
                "stamp_tax_in": 0.0002,
                "stamp_tax_out": 0.0002,
                "commission": 5,
            }
            # fake_data显示设置
            showconfig = {
                "range_low": -10,
                "range_high": 11,
                "range_eff": 0.01,
                "mount_low": -4,
                "mount_high": 6,
                "mount_eff": 0.2,
            }
            # showconfig = {
            #     "range_low": -3,
            #     "range_high": 4,
            #     "range_eff": 0.01,
            #     "mount_low": -1,
            #     "mount_high": 2,
            #     "mount_eff": 0.5,
            # }
            backtest.simulate_lastday(para_config, showconfig)
        else:
            raise Exception("func_type 只能是 train, backtest, lastday")


def main(paralist):
    logger.info(paralist)
    account_list = [
        {
            "account": 1,
            "back_test": {
                "test_type": "实盘",
                "start_train": "19900101",
                "end_train": "19900101",
                "start_predict": "19900101",
                "end_predict": "19900101",
                "heartbeat": 0.0,
                "initial_capital": 10000.0,
            },
            "data_ori": {
                "data_type": "实盘",
                "csv_dir": data_path,
                "symbol_list": ["SAPower", "DalianRP", "ChinaBank"],
                # "symbol_list": ["SAPower"],
                "ave_list": [1, 3, 5, 7, 17, 20, 23, 130, 140, 150],
                "bband_list": [5, 19, 37],
                "ret_list": [1, 3, 5, 7, 17, 20, 23, 130, 140, 150],
            },
            "stratgey": {
                "stratgey_name": "cross_break",
            },
            "portfolio": {
                "portfolio_name": None
            },
        },
        {
            "account": 2,
            "desc": "非tushare,离线测试",
            "back_test": {
                "test_type": "模拟",
                "start_train": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "end_train": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "start_predict": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "end_predict": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "heartbeat": 0.0,
                "initial_capital": 10000.0,
            },
            "data_ori": {
                "data_type": "模拟",
                # "data_type": "实盘",
                "csv_dir": data_path,
                # "symbol_list": ["SAPower", "DalianRP", "ChinaBank"],
                # "symbol_list": ["SAPower", "DalianRP", "ChinaBank"],
                "symbol_list": ["SAPower"],
                "ave_list": [1, 3, 5, 11, 19, 37, 67],
                "bband_list": [5, 19, 37],
                "ret_list": [1, 3, 5, 7, 17, 20, 23, 130, 140, 150],
            },
            "stratgey": {
                "stratgey_name": "cross_break",
            },
            "portfolio": {
                "portfolio_name": None
            },
        },
        {
            "account": 3,
            "desc": "tushare,离线测试",
            "back_test": {
                "test_type": "模拟",
                "start_train": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "end_train": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "start_predict": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "end_predict": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "heartbeat": 0.0,
                "initial_capital": 10000.0,
            },
            "data_ori": {
                # "func_type": "lastday",
                "func_type": "train",
                # "func_type": "backtest",
                # "func_type": "predict",
                "data_type": "模拟",
                # "data_type": "网络",
                # "data_type": "实盘",
                "csv_dir": data_path,
                # "symbol_list": ["SAPower", "DalianRP", "ChinaBank"],
                # "symbol_list": ["DalianRP"],
                # "symbol_list": ["SAPower"],
                # "symbol_list": ["SAPower", "DalianRP"],
                # "symbol_list": ["ChinaBank"],
                "symbol_list": ["000001_D", "000002_D"],
                # "symbol_list": ["000002_D"],
                "ave_list": [1, 3, 5, 11, 19, 37, 67],
                "bband_list": [1],
                # "bband_list": [5],
                # "bband_list": [19],
                # "bband_list": [37],
                # "bband_list": [1, 5, 19],
                # "bband_list": [5, 19, 37],
                "ret_list": [1, 3, 5, 7, 17, 20, 23, 130, 140, 150],
            },
            "stratgey": {
                "stratgey_name": "cross_break",
            },
            "portfolio": {
                "portfolio_name": None
            },
        }

    ]
    ins = Acount(account_list[2])
    ins()


if __name__ == "__main__":
    logger.info("".center(100, "*"))
    logger.info("welcome to surfing".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    main(sys.argv[1:])
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
