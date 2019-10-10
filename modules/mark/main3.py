import sys, os
import datetime
import json
import itertools
from modules.portfolio import Portfolio
from modules.event import *
from modules.datahandle import CSVDataHandler, CSVAppendDataHandler, LoadCSVHandler
from modules.strategys import MovingAverageCrossStrategy, MultiCrossStrategy, MlaStrategy
from modules.executions import SimulatedExecutionHandler
from modules.backtests import Backtest, LoadBacktest
from utils.log_tool import *
import pyttsx3
import tushare as ts


def choice_list(plate_list):
    # 行业分类
    tt = ts.get_industry_classified()
    infos = tt['code'].groupby([tt['c_name']]).apply(list)
    industjson = json.loads(infos.to_json(orient='index', force_ascii=False))
    # 中小板分类
    tt = ts.get_sme_classified()
    allsmartlist = []
    for plate1 in plate_list:
        allsmartlist.append(list(set(tt["code"]).intersection(set(industjson[plate1]))))
    # allsmartlist = list(set(itertools.chain(*allsmartlist)))
    allsmartlist = [i1 + "_D" for i1 in set(itertools.chain(*allsmartlist))]
    # smartlist = list(set(tt["code"]).intersection(set(industjson["电子信息"])))
    # # 概念分类
    # tt = ts.get_concept_classified()
    # infos = tt['code'].groupby([tt['c_name']]).apply(to_list)
    # conceptjson = json.loads(infos.to_json(orient='index', force_ascii=False))
    # # 小轻，突破边缘的信息类
    # nearbreaklist = list(set(smartlist).intersection(set(conceptjson["智能机器"])))
    # print(conceptjson["智能机器"])
    # print(nearbreaklist)
    return allsmartlist


class Acount(object):
    def __init__(self, config):
        self.account = config["account"]
        self.func_type = config["data_ori"]["func_type"]
        self.test_type = config["back_test"]["test_type"]
        self.start_train = config["back_test"]["start_train"]
        self.end_train = config["back_test"]["end_train"]
        self.end_predict = config["back_test"]["end_predict"]
        self.initial_capital = config["back_test"]["initial_capital"]
        self.heartbeat = config["back_test"]["heartbeat"]
        self.start_predict = config["data_ori"]["start_predict"]
        self.date_range = config["data_ori"]["date_range"]
        self.data_type = config["data_ori"]["data_type"]
        self.bband_list = config["data_ori"]["bband_list"]
        self.split = config["data_ori"]["split"]
        self.newdata = config["data_ori"]["newdata"]
        self.csv_dir = config["data_ori"]["csv_dir"]
        self.plate_list = config["data_ori"]["plate_list"]
        self.symbol_list = config["data_ori"]["symbol_list"]
        self.oper_num = config["data_ori"]["oper_num"]
        self.ave_list = config["data_ori"]["ave_list"]
        self.bband_list = config["data_ori"]["bband_list"]
        self.stratgey_name = config["stratgey"]["stratgey_name"]
        self.portfolio_name = config["portfolio"]["portfolio_name"]
        self.email_list = config["assist_option"]["email_list"]
        self.para_config = config["para_config"]
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
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list,
                    CSVDataHandler, SimulatedExecutionHandler, Portfolio, MovingAverageCrossStrategy)
            elif self.data_type == "实盘":  # 已有数据，动态模拟, 未完善
                backtest = Backtest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list,
                    CSVAppendDataHandler, SimulatedExecutionHandler, Portfolio, MultiCrossStrategy)
            elif self.data_type == "symbol_train_type":  # 已有数据，直观统计
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=1, date_range=self.date_range, oper_num=self.oper_num, assistant=self.email_list)
            elif self.data_type == "general_train_type":  # 已有数据，直观统计
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self._get_train_list(), self.ave_list, self.bband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=0, date_range=self.date_range, oper_num=self.oper_num, assistant=self.email_list)
            elif self.data_type == "plate_train_type":  # 已有数据，直观统计
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, choice_list(self.plate_list), self.ave_list, self.bband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=1, date_range=self.date_range, oper_num=self.oper_num, assistant=self.email_list)
            elif self.data_type == "网络获取数据":  # 已有数据，统计强化学习
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    None, self._get_train_list(), self.ave_list, self.bband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=1, date_range=self.date_range, oper_num=self.oper_num, assistant=self.email_list)
                return None
            else:
                raise Exception("error data_type 只允许：实盘demo, 实盘, 模拟, 网络")
        else:
            raise Exception("error test type.")
        # 2. 判断执行功能
        if self.func_type == "train":
            backtest.train()
        elif self.func_type == "backtest":
            backtest.simulate_trading(self.para_config, startdate=self.start_predict)
        elif self.func_type == "lastday":
            # fake_data显示设置
            showconfig = {
                "range_low": -10,
                "range_high": 11,
                # "range_low": -1,
                # "range_high": 2,
                "range_eff": 0.01,
                # "mount_low": -4,
                # "mount_high": 6,
                # "mount_eff": 0.2,
                "mount_low": -1,
                "mount_high": 1,
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
            backtest.simulate_lastday(self.para_config, showconfig, startdate=self.start_predict)
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
                "ave_list": [1, 3, 5, 7, 17, 20, 23, 130, 140, 150],
                "bband_list": [5, 19, 37],
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
                "symbol_list": ["SAPower"],
                "ave_list": [1, 3, 5, 11, 19, 37, 67],
                "bband_list": [5, 19, 37],
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
                "end_predict": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "heartbeat": 0.0,
                "initial_capital": 10000.0,
            },
            "data_ori": {
                # "func_type": "train",
                # "data_type": "general_train_type",
                # "date_range": [0, None],
                "split": 0.8,
                # 使用已下好的原生数据
                "newdata": 1,
                # "func_type": "train",
                # "data_type": "plate_train_type",
                # "date_range": [0, None],
                "plate_list": ["电子信息"],

                # "func_type": "train",
                "func_type": "backtest",
                "data_type": "symbol_train_type",
                "date_range": [-10, None],
                # "func_type": "lastday",
                # "data_type": "网络获取数据",
                "start_predict": "2019-09-29 00:00:00",
                # "start_predict": None,
                # "date_range": [0, None],
                # "data_type": "实盘",
                "csv_dir": data_path,
                "symbol_list": ["000001_D", "000002_D"],
                # "symbol_list": ["000002_D"],
                "oper_num": 3,
                "ave_list": [1, 3, 5, 11, 19, 37, 67],
                # "bband_list": [1],
                # "bband_list": [5],
                # "bband_list": [19],
                # "bband_list": [37],
                "bband_list": [1, 5],
                # "bband_list": [1, 2, 3, 4, 5, 6, 7],
                # "bband_list": [5, 19],
                # "bband_list": [1, 5, 19],
                # "bband_list": [1, 5, 19, 37],
                # "bband_list": [5, 19, 37],
            },
            "stratgey": {
                "stratgey_name": "cross_break",
            },
            "portfolio": {
                "portfolio_name": None
            },
            "assist_option": {
                # "email_list": ["a1593572007@126.com", "619041014@qq.com"],
                "email_list": ["a1593572007@126.com"],
            },
            "para_config": {
                "hand_unit": 100,
                "initial_capital": 10000.0,
                "stamp_tax_in": 0.0,
                "stamp_tax_out": 0.001,
                "commission": 5,
                "commission_rate": 0.0003,
            }
        }

    ]
    ins = Acount(account_list[2])
    ins()


def test():
    code = "000001"
    startdate = "2019-09-29 00:00:00"
    df2 = ts.get_hist_data(code, ktype="D", start=startdate)
    print(df2)
    df2 = ts.get_realtime_quotes(["000001"])[["date", "open", "high", "low", "price", "volume"]]
    df2 = df2.rename(columns={"price": "close"})
    print(df2)
    exit()


if __name__ == "__main__":
    # test()
    engine = pyttsx3.init()
    logger.info("".center(100, "*"))
    logger.info("welcome to surfing".center(30, " ").center(100, "*"))
    engine.setProperty('rate', int(engine.getProperty('rate') * 0.85))
    engine.setProperty('volume', engine.getProperty('volume') * 1.0)
    engine.say("welcome to surfing!")
    engine.runAndWait()
    logger.info("".center(100, "*"))
    logger.info("")
    main(sys.argv[1:])
    logger.info("")
    engine.say("任务完成。")
    engine.say("bye!")
    engine.runAndWait()
    logger.info("bye!".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
