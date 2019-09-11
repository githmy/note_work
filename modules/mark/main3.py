import sys
import datetime
from modules.portfolio import Portfolio
from modules.event import *
from modules.datahandle import CSVDataHandler, CSVAppendDataHandler, LoadCSVHandler
from modules.strategys import MovingAverageCrossStrategy, MultiCrossStrategy, MlaStrategy
from modules.executions import SimulatedExecutionHandler
from modules.backtests import Backtest, LoadBacktest
from utils.log_tool import *


class Acount(object):
    def __init__(self, config):
        self.account = config["account"]
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

    def __call__(self, *args, **kwargs):
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
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy)
            # elif self.data_type == "学习":  # 已有数据，统计强化学习
            #     backtest = Backtest(
            #         self.initial_capital, self.heartbeat, self.start_predict,
            #         self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.ret_list,
            #         CSVAppendDataHandler, SimulatedExecutionHandler, Portfolio, MultiCrossStrategy)
            else:
                raise Exception("error data_type.")
        else:
            raise Exception("error test type.")
        backtest.simulate_trading()


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
        }
    ]
    ins = Acount(account_list[1])
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
