from __future__ import print_function
import os
import numpy as np
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
from pyalgotrade import strategy
import itertools
from utils.log_tool import *
from utils.path_tool import makesurepath
from modules.event import *
from modules.stocks.stock_network2 import CRNN, CRNNevery
from modules.stocks.finance_tool import TradeTool
import simplejson
import argparse


def parseArgs(args):
    parser = argparse.ArgumentParser()
    globalArgs = parser.add_argument_group('Global options')
    globalArgs.add_argument('--modelname', default=None, nargs='?',
                            choices=["full", "one", "one_y", "one_space", "one_attent", "one_attent60"])
    globalArgs.add_argument('--learnrate', type=float, nargs='?', default=None)
    globalArgs.add_argument('--globalstep', type=int, nargs='?', default=None)
    globalArgs.add_argument('--dropout', type=float, nargs='?', default=None)
    globalArgs.add_argument('--normal', type=float, nargs='?', default=None)
    globalArgs.add_argument('--sub_fix', type=str, nargs='?', default=None)
    return parser.parse_args(args)


def print_range(train_bars, bband_list, ave_list, data_range, split):
    totallenth = len(train_bars.symbol_ori_data[train_bars.symbol_list[0]]["close"])
    mid_lenth = max(bband_list[-1] - 1, ave_list[-1])
    trainpre_pos = max(data_range[0] if data_range[0] is not None else 0, ave_list[-1] - 1)
    validaft_lenth = max(data_range[1] if data_range[1] is not None else 0, bband_list[-1])
    usefull_lenth = totallenth - trainpre_pos - validaft_lenth - mid_lenth
    trainaft_pos = int(usefull_lenth * split) + trainpre_pos
    validpre_pos = trainpre_pos + int(usefull_lenth * split) + mid_lenth
    validaft_pos = trainpre_pos + usefull_lenth + mid_lenth
    print("total length: {} train range:{}-{}. valid range:{}-{}.".format(totallenth, trainpre_pos, trainaft_pos,
                                                                          validpre_pos, validaft_pos))


# 封装对数据的计算，并且生成相应的信号:  策略处理基类，可用于处理历史和实际交易数据，只需把数据存到队列中。
# 移动平均跨越策略。用短期/长期移动平均值进行基本的移动平均跨越的实现。
class MovingAverageCrossStrategy(strategy.BacktestingStrategy):
    """
    封装对数据的计算，并且生成相应的信号:  策略处理基类，可用于处理历史和实际交易数据，只需把数据存到队列中。
    移动平均跨越策略。用短期/长期移动平均值进行基本的移动平均跨越的实现。
    """

    def __init__(self, bars, events, ave_list, short_window=100, long_window=400):
        """
        Initialises the buy and hold strategy.

        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.ave_list = ave_list
        self.short_window = short_window
        self.long_window = long_window

        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def onBars(self, bars):
        # 交易规则
        account = self.getBroker().getCash()
        bar = bars[self.__instrument]
        if self.__position is None:
            one = bar.getPrice() * 100
            oneUnit = account // one
            if oneUnit > 0 and self.__diff[-1] > self.__break:
                self.__position = self.enterLong(self.__instrument, oneUnit * 100, True)
        elif self.__diff[-1] < self.__withdown and not self.__position.exitActive():
            self.__position.exitMarket()
            # # SMA的计算存在窗口，所以前面的几个bar下是没有SMA的数据的.
            # if self.__sma[-1] is None:
            #     return
            #     # bar.getTyoicalPrice = (bar.getHigh() + bar.getLow() + bar.getClose())/ 3.0
            #
            # bar = bars[self.__instrument]
            # # If a position was not opened, check if we should enter a long position.
            # if self.__position is None:  # 如果手上没有头寸，那么
            #     if bar.getPrice() > self.__sma[-1]:
            #         # 开多，如果现价大于移动均线，且当前没有头寸.
            #         self.__position = self.enterLong(self.__instrument, 100, True)
            #         # 当前有多头头寸，平掉多头头寸.
            # elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():
            #     self.__position.exitMarket()

    # bought dict 添加key,并对所有代码设置为out
    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    # 基于MAC,SMA生成一组新的信号，进入market就是短期移动平均超过长期移动平均
    def calculate_signals(self, event):
        """
        基于MAC,SMA生成一组新的信号，进入market就是短期移动平均超过长期移动平均
        Parameters
        event - A Bar object. 
        """
        if event.type == 'BAR':
            for symbol in self.symbol_list:
                bars = self.bars.get_latest_bars_values(symbol, "close", N=self.long_window)
                if bars is not None and bars != []:
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])
                    # 最新的bar时间
                    dt = self.bars.get_latest_bar_datetime(symbol)
                    sig_dir = ""
                    strength = 1.0
                    strategy_id = 1
                    # 判断信号类型
                    if short_sma > long_sma and self.bought[symbol] == "OUT":
                        sig_dir = 'LONG'
                        # 信号 生成
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'
                    elif short_sma < long_sma and self.bought[symbol] == "LONG":
                        sig_dir = 'EXIT'
                        # 信号 生成
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'OUT'


class MultiCrossStrategy(strategy.BacktestingStrategy):
    """
    各种交叉线和信号策略
    """

    def __init__(self, bars, events, ave_list, short_window=100, long_window=400):
        """
        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.ave_list = ave_list
        self.short_window = short_window
        self.long_window = long_window

        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def onBars(self, bars):
        # 交易规则
        account = self.getBroker().getCash()
        bar = bars[self.__instrument]
        if self.__position is None:
            one = bar.getPrice() * 100
            oneUnit = account // one
            if oneUnit > 0 and self.__diff[-1] > self.__break:
                self.__position = self.enterLong(self.__instrument, oneUnit * 100, True)
        elif self.__diff[-1] < self.__withdown and not self.__position.exitActive():
            self.__position.exitMarket()

    # bought dict 添加key,并对所有代码设置为out
    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event):
        """
        基于一组均线生成一组新的信号，供训练用
        """
        curv_list = list(itertools.combinations(self.ave_list, 2))
        if event.type == 'BAR':
            for symbol in self.symbol_list:
                bars = self.bars.get_latest_bars_values(symbol, "close", N=curv_list[-1][1])
                if bars is not None and bars != []:
                    # 最新的bar时间
                    dt = self.bars.get_latest_bar_datetime(symbol)
                    for pair in curv_list:
                        pre_short_sma = np.mean(bars[-(pair[0] + 1):-1])
                        pre_long_sma = np.mean(bars[-(pair[1] + 1):-1])
                        short_sma = np.mean(bars[-pair[0]:])
                        long_sma = np.mean(bars[-pair[1]:])
                        sig_dir = 0
                        strength = 1.0
                        strategy_id = 1
                        # 判断信号类型[-2,-1,0,1,2] [一直小于，下穿，等于，上穿，一直大于]
                        if short_sma > long_sma and pre_short_sma > pre_long_sma:
                            sig_dir = 2
                        elif short_sma > long_sma and pre_short_sma <= pre_long_sma:
                            sig_dir = 1
                        elif short_sma < long_sma and pre_short_sma >= pre_long_sma:
                            sig_dir = -1
                        elif short_sma < long_sma and pre_short_sma < pre_long_sma:
                            sig_dir = -2
                        else:
                            return None
                        signal = SignalTrainEvent(strategy_id, symbol, dt, list(pair).append(sig_dir), strength)
                        self.events.put(signal)


class MlaStrategy(strategy.BacktestingStrategy):
    """各种交叉线和信号策略"""

    def __init__(self, bars, events, ave_list, bband_list):
        """
        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.ave_list = ave_list
        self.bband_list = bband_list

        self.toolins = TradeTool()
        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()
        self.trainconfig = {}

    def onBars(self, bars):
        # 交易规则
        account = self.getBroker().getCash()
        bar = bars[self.__instrument]
        if self.__position is None:
            one = bar.getPrice() * 100
            oneUnit = account // one
            if oneUnit > 0 and self.__diff[-1] > self.__break:
                self.__position = self.enterLong(self.__instrument, oneUnit * 100, True)
        elif self.__diff[-1] < self.__withdown and not self.__position.exitActive():
            self.__position.exitMarket()

    # bought dict 添加key,并对所有代码设置为out
    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_probability_signals(self, event):
        """
        kelly_formula
        """
        self.system_risk = 0.0001  # 系统归零概率
        self.system_move = 1  # 上下概率平移系数
        self.system_ram_vari = 1  # 振幅系数
        self.bars.upprb = {}  # 扣除系统归零概率后的向上概率
        self.bars.downprb = {}  # 扣除系统归零概率后的向上概率
        self.bars.f_ratio = {}
        self.bars.gain = {}
        for s in self.bars.symbol_list:
            self.bars.upprb[s] = []
            self.bars.downprb[s] = []
            self.bars.f_ratio[s] = []
            self.bars.gain[s] = []
            for id1, aven in enumerate(self.bband_list):
                fixratio = self.system_ram_vari / aven
                fw = self.bars.symbol_aft_retp_high[s][id1] - 1.0
                fw = fw * fixratio
                fl = 1.0 - self.bars.symbol_aft_retp_low[s][id1]
                fl = fl * fixratio
                upconst = np.exp(-(fw * self.system_move) ** 2 / self.bars.symbol_aft_half_std_up[s][id1] ** 2)
                downconst = np.exp(-(fl / self.system_move) ** 2 / self.bars.symbol_aft_half_std_down[s][id1] ** 2)
                # 加入系统风险后的极值一阶导数方程： y = a*f_ratio^2+b*f_ratio+c
                p = (1 - self.system_risk) * upconst / (upconst + downconst)
                q = (1 - self.system_risk) * downconst / (upconst + downconst)
                wm = self.toolins.kari_fix_normal_w(p, q, fw, fl, self.system_risk)
                wm[:][wm[:] < 0] = 0
                self.bars.upprb[s].append(p)
                self.bars.downprb[s].append(q)
                self.bars.f_ratio[s].append(wm)
                self.bars.gain[s].append(self.toolins.kari_fix_normal_g(p, q, fw, fl, self.system_risk, wm))

    def _prepare_model_para(self, model_paras=None, args=None):
        # 2. 模型参数赋值
        config = {}
        parafile = os.path.join("config", "para.json")
        argspar = parseArgs(args)
        if argspar.modelname is None:
            # hpara = simplejson.load(open(parafile))
            hpara = model_paras
            config["modelname"] = hpara["model"]["modelname"]
            config["sub_fix"] = hpara["model"]["sub_fix"]
            config["tailname"] = "%s-%s" % (config["modelname"], config["sub_fix"])
        else:
            config["modelname"] = argspar.modelname
            if argspar.sub_fix is None:
                config["tailname"] = "%s-" % (argspar.modelname)
            else:
                config["sub_fix"] = argspar.sub_fix
                config["tailname"] = "%s-%s" % (argspar.modelname, argspar.sub_fix)
            parafile = "para_%s.json" % (config["tailname"])
            hpara = simplejson.load(open(parafile))
            # if argspar.sub_fix is None:
            #     config["sub_fix"] = hpara["model"]["sub_fix"]
        config["early_stop"] = hpara["env"]["early_stop"]
        if argspar.learnrate is None:
            config["learn_rate"] = hpara["env"]["learn_rate"]
        else:
            config["learn_rate"] = argspar.learnrate

        if argspar.globalstep is None:
            globalstep = hpara["model"]["globalstep"]
        else:
            globalstep = argspar.globalstep
        if argspar.dropout is None:
            config["dropout"] = hpara["model"]["dropout"]
        else:
            config["dropout"] = argspar.dropout
        if argspar.normal is None:
            config["normal"] = hpara["model"]["normal"]
        else:
            config["normal"] = argspar.normal

        config["scope"] = hpara["env"]["scope"]
        config["inputdim"] = hpara["env"]["inputdim"]
        config["outspace"] = hpara["env"]["outspace"]
        config["single_num"] = hpara["env"]["single_num"]
        config["modelfile"] = hpara["model"]["file"]
        config["retrain"] = hpara["model"]["retrain"]
        config["batchsize"] = hpara["env"]["batch_size"]
        config["epoch"] = hpara["env"]["epoch"]
        print()
        print("**********************************************************")
        print("parafile:", parafile)
        print("modelname:", config["modelname"])
        print("tailname:", config["tailname"])
        print("learn_rate:", config["learn_rate"])
        print("dropout:", config["dropout"])
        print("**********************************************************")
        self.trainconfig = config

    def _prepare_every_train_data(self, train_bars, ave_list, bband_list, data_range, split=0.8):
        mult_charact_trainx = []
        mult_charact_validx = []
        mult_charact_trainy_ret_a = []
        mult_charact_trainy_ret_h = []
        mult_charact_trainy_ret_l = []
        mult_charact_trainy_stdup = []
        mult_charact_trainy_stddw = []
        mult_charact_trainy_drawup = []
        mult_charact_trainy_drawdw = []
        mult_charact_validy_ret_a = []
        mult_charact_validy_ret_h = []
        mult_charact_validy_ret_l = []
        mult_charact_validy_stdup = []
        mult_charact_validy_stddw = []
        mult_charact_validy_drawup = []
        mult_charact_validy_drawdw = []
        symbol_list = list(train_bars.symbol_pre_half_std_up.keys())
        totallenth = len(train_bars.symbol_ori_data[symbol_list[0]]["close"])
        mid_lenth = max(bband_list[-1] - 1, ave_list[-1])
        trainpre_pos = max(data_range[0] if data_range[0] is not None else 0, ave_list[-1] - 1)
        validaft_lenth = max(data_range[1] if data_range[1] is not None else 0, bband_list[-1])
        usefull_lenth = totallenth - trainpre_pos - validaft_lenth - mid_lenth
        trainaft_pos = int(usefull_lenth * split) + trainpre_pos
        validpre_pos = trainpre_pos + int(usefull_lenth * split) + mid_lenth
        validaft_pos = trainpre_pos + usefull_lenth + mid_lenth
        print("total length: {} train range:{}-{}. valid range:{}-{}.".format(totallenth, trainpre_pos, trainaft_pos,
                                                                              validpre_pos, validaft_pos))
        for s in symbol_list:
            # 1. 加载标签数据
            xchara_trainlist = []
            xchara_validlist = []
            xlen_slist = len(ave_list)
            for single_chara in range(xlen_slist):
                xchara_trainlist.append(train_bars.symbol_pre_half_std_up[s][single_chara][trainpre_pos:trainaft_pos])
                xchara_trainlist.append(train_bars.symbol_pre_half_std_down[s][single_chara][trainpre_pos:trainaft_pos])
                xchara_validlist.append(train_bars.symbol_pre_half_std_up[s][single_chara][validpre_pos:validaft_pos])
                xchara_validlist.append(train_bars.symbol_pre_half_std_down[s][single_chara][validpre_pos:validaft_pos])
                for single2_chara in range(xlen_slist):
                    xchara_trainlist.append(
                        train_bars.symbol_pre_retp[s][single_chara][single2_chara][trainpre_pos:trainaft_pos])
                    xchara_trainlist.append(
                        train_bars.symbol_pre_retm[s][single_chara][single2_chara][trainpre_pos:trainaft_pos])
                    xchara_validlist.append(
                        train_bars.symbol_pre_retp[s][single_chara][single2_chara][validpre_pos:validaft_pos])
                    xchara_validlist.append(
                        train_bars.symbol_pre_retm[s][single_chara][single2_chara][validpre_pos:validaft_pos])
            tmp_xtrainnp = np.vstack(xchara_trainlist)
            tmp_xvalidnp = np.vstack(xchara_validlist)
            # aft
            ychara_ret_a_trainlist = []
            ychara_ret_h_trainlist = []
            ychara_ret_l_trainlist = []
            ychara_stdup_trainlist = []
            ychara_stddw_trainlist = []
            ychara_drawup_trainlist = []
            ychara_drawdw_trainlist = []
            ychara_ret_a_validlist = []
            ychara_ret_h_validlist = []
            ychara_ret_l_validlist = []
            ychara_stdup_validlist = []
            ychara_stddw_validlist = []
            ychara_drawup_validlist = []
            ychara_drawdw_validlist = []
            ylen_slist = len(bband_list)
            for single_chara in range(ylen_slist):
                ychara_ret_a_trainlist.append(train_bars.symbol_aft_reta[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_ret_h_trainlist.append(
                    train_bars.symbol_aft_retp_high[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_ret_l_trainlist.append(
                    train_bars.symbol_aft_retp_low[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_stdup_trainlist.append(
                    train_bars.symbol_aft_half_std_up[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_stddw_trainlist.append(
                    train_bars.symbol_aft_half_std_down[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_drawup_trainlist.append(train_bars.symbol_aft_drawup[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_drawdw_trainlist.append(
                    train_bars.symbol_aft_drawdown[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_ret_a_validlist.append(train_bars.symbol_aft_reta[s][single_chara][validpre_pos:validaft_pos])
                ychara_ret_h_validlist.append(
                    train_bars.symbol_aft_retp_high[s][single_chara][validpre_pos:validaft_pos])
                ychara_ret_l_validlist.append(
                    train_bars.symbol_aft_retp_low[s][single_chara][validpre_pos:validaft_pos])
                ychara_stdup_validlist.append(
                    train_bars.symbol_aft_half_std_up[s][single_chara][validpre_pos:validaft_pos])
                ychara_stddw_validlist.append(
                    train_bars.symbol_aft_half_std_down[s][single_chara][validpre_pos:validaft_pos])
                ychara_drawup_validlist.append(train_bars.symbol_aft_drawup[s][single_chara][validpre_pos:validaft_pos])
                ychara_drawdw_validlist.append(
                    train_bars.symbol_aft_drawdown[s][single_chara][validpre_pos:validaft_pos])
            tmp_xtrainnp = np.transpose(tmp_xtrainnp)
            tmp_xvalidnp = np.transpose(tmp_xvalidnp)
            mult_charact_trainx.append(tmp_xtrainnp)
            mult_charact_validx.append(tmp_xvalidnp)
            tmp_ychara_ret_a_trainlist = np.transpose(np.vstack(ychara_ret_a_trainlist))
            tmp_ychara_ret_h_trainlist = np.transpose(np.vstack(ychara_ret_h_trainlist))
            tmp_ychara_ret_l_trainlist = np.transpose(np.vstack(ychara_ret_l_trainlist))
            tmp_ychara_stdup_trainlist = np.transpose(np.vstack(ychara_stdup_trainlist))
            tmp_ychara_stddw_trainlist = np.transpose(np.vstack(ychara_stddw_trainlist))
            tmp_ychara_drawup_trainlist = np.transpose(np.vstack(ychara_drawup_trainlist))
            tmp_ychara_drawdw_trainlist = np.transpose(np.vstack(ychara_drawdw_trainlist))
            tmp_ychara_ret_a_validlist = np.transpose(np.vstack(ychara_ret_a_validlist))
            tmp_ychara_ret_h_validlist = np.transpose(np.vstack(ychara_ret_h_validlist))
            tmp_ychara_ret_l_validlist = np.transpose(np.vstack(ychara_ret_l_validlist))
            tmp_ychara_stdup_validlist = np.transpose(np.vstack(ychara_stdup_validlist))
            tmp_ychara_stddw_validlist = np.transpose(np.vstack(ychara_stddw_validlist))
            tmp_ychara_drawup_validlist = np.transpose(np.vstack(ychara_drawup_validlist))
            tmp_ychara_drawdw_validlist = np.transpose(np.vstack(ychara_drawdw_validlist))
            mult_charact_trainy_ret_a.append(tmp_ychara_ret_a_trainlist)
            mult_charact_trainy_ret_h.append(tmp_ychara_ret_h_trainlist)
            mult_charact_trainy_ret_l.append(tmp_ychara_ret_l_trainlist)
            mult_charact_trainy_stdup.append(tmp_ychara_stdup_trainlist)
            mult_charact_trainy_stddw.append(tmp_ychara_stddw_trainlist)
            mult_charact_trainy_drawup.append(tmp_ychara_drawup_trainlist)
            mult_charact_trainy_drawdw.append(tmp_ychara_drawdw_trainlist)
            mult_charact_validy_ret_a.append(tmp_ychara_ret_a_validlist)
            mult_charact_validy_ret_h.append(tmp_ychara_ret_h_validlist)
            mult_charact_validy_ret_l.append(tmp_ychara_ret_l_validlist)
            mult_charact_validy_stdup.append(tmp_ychara_stdup_validlist)
            mult_charact_validy_stddw.append(tmp_ychara_stddw_validlist)
            mult_charact_validy_drawup.append(tmp_ychara_drawup_validlist)
            mult_charact_validy_drawdw.append(tmp_ychara_drawdw_validlist)
        all_xtrainnp = np.vstack(mult_charact_trainx)
        all_xvalidnp = np.vstack(mult_charact_validx)
        all_ytrainnp_ret_a = np.vstack(mult_charact_trainy_ret_a)
        all_ytrainnp_ret_h = np.vstack(mult_charact_trainy_ret_h)
        all_ytrainnp_ret_l = np.vstack(mult_charact_trainy_ret_l)
        all_ytrainnp_stdup = np.vstack(mult_charact_trainy_stdup)
        all_ytrainnp_stddw = np.vstack(mult_charact_trainy_stddw)
        all_ytrainnp_drawup = np.vstack(mult_charact_trainy_drawup)
        all_ytrainnp_drawdw = np.vstack(mult_charact_trainy_drawdw)
        all_yvalidnp_ret_a = np.vstack(mult_charact_validy_ret_a)
        all_yvalidnp_ret_h = np.vstack(mult_charact_validy_ret_h)
        all_yvalidnp_ret_l = np.vstack(mult_charact_validy_ret_l)
        all_yvalidnp_stdup = np.vstack(mult_charact_validy_stdup)
        all_yvalidnp_stddw = np.vstack(mult_charact_validy_stddw)
        all_yvalidnp_drawup = np.vstack(mult_charact_validy_drawup)
        all_yvalidnp_drawdw = np.vstack(mult_charact_validy_drawdw)
        # 4. 处理nan inf
        all_xtrainnp[:, :][np.isnan(all_xtrainnp[:, :])] = 0
        all_xtrainnp[:, :][np.isinf(all_xtrainnp[:, :])] = 0
        all_xvalidnp[:, :][np.isnan(all_xvalidnp[:, :])] = 0
        all_xvalidnp[:, :][np.isinf(all_xvalidnp[:, :])] = 0
        all_ytrainnp_ret_a = np.array(all_ytrainnp_ret_a.tolist())
        all_ytrainnp_ret_a[:, :][np.isnan(all_ytrainnp_ret_a[:, :])] = 0
        all_ytrainnp_ret_a[:, :][np.isinf(all_ytrainnp_ret_a[:, :])] = 0
        all_ytrainnp_ret_h = np.array(all_ytrainnp_ret_h.tolist())
        all_ytrainnp_ret_h[:, :][np.isnan(all_ytrainnp_ret_h[:, :])] = 0
        all_ytrainnp_ret_h[:, :][np.isinf(all_ytrainnp_ret_h[:, :])] = 0
        all_ytrainnp_ret_l = np.array(all_ytrainnp_ret_l.tolist())
        all_ytrainnp_ret_l[:, :][np.isnan(all_ytrainnp_ret_l[:, :])] = 0
        all_ytrainnp_ret_l[:, :][np.isinf(all_ytrainnp_ret_l[:, :])] = 0
        all_ytrainnp_stdup = np.array(all_ytrainnp_stdup.tolist())
        all_ytrainnp_stdup[:, :][np.isnan(all_ytrainnp_stdup[:, :])] = 0
        all_ytrainnp_stdup[:, :][np.isinf(all_ytrainnp_stdup[:, :])] = 0
        all_ytrainnp_stddw = np.array(all_ytrainnp_stddw.tolist())
        all_ytrainnp_stddw[:, :][np.isnan(all_ytrainnp_stddw[:, :])] = 0
        all_ytrainnp_stddw[:, :][np.isinf(all_ytrainnp_stddw[:, :])] = 0
        all_ytrainnp_drawup = np.array(all_ytrainnp_drawup.tolist())
        all_ytrainnp_drawup[:, :][np.isnan(all_ytrainnp_drawup[:, :])] = 0
        all_ytrainnp_drawup[:, :][np.isinf(all_ytrainnp_drawup[:, :])] = 0
        all_ytrainnp_drawdw = np.array(all_ytrainnp_drawdw.tolist())
        all_ytrainnp_drawdw[:, :][np.isnan(all_ytrainnp_drawdw[:, :])] = 0
        all_ytrainnp_drawdw[:, :][np.isinf(all_ytrainnp_drawdw[:, :])] = 0
        all_yvalidnp_ret_a = np.array(all_yvalidnp_ret_a.tolist())
        all_yvalidnp_ret_a[:, :][np.isnan(all_yvalidnp_ret_a[:, :])] = 0
        all_yvalidnp_ret_a[:, :][np.isinf(all_yvalidnp_ret_a[:, :])] = 0
        all_yvalidnp_ret_h = np.array(all_yvalidnp_ret_h.tolist())
        all_yvalidnp_ret_h[:, :][np.isnan(all_yvalidnp_ret_h[:, :])] = 0
        all_yvalidnp_ret_h[:, :][np.isinf(all_yvalidnp_ret_h[:, :])] = 0
        all_yvalidnp_ret_l = np.array(all_yvalidnp_ret_l.tolist())
        all_yvalidnp_ret_l[:, :][np.isnan(all_yvalidnp_ret_l[:, :])] = 0
        all_yvalidnp_ret_l[:, :][np.isinf(all_yvalidnp_ret_l[:, :])] = 0
        all_yvalidnp_stdup = np.array(all_yvalidnp_stdup.tolist())
        all_yvalidnp_stdup[:, :][np.isnan(all_yvalidnp_stdup[:, :])] = 0
        all_yvalidnp_stdup[:, :][np.isinf(all_yvalidnp_stdup[:, :])] = 0
        all_yvalidnp_stddw = np.array(all_yvalidnp_stddw.tolist())
        all_yvalidnp_stddw[:, :][np.isnan(all_yvalidnp_stddw[:, :])] = 0
        all_yvalidnp_stddw[:, :][np.isinf(all_yvalidnp_stddw[:, :])] = 0
        all_yvalidnp_drawup = np.array(all_yvalidnp_drawup.tolist())
        all_yvalidnp_drawup[:, :][np.isnan(all_yvalidnp_drawup[:, :])] = 0
        all_yvalidnp_drawup[:, :][np.isinf(all_yvalidnp_drawup[:, :])] = 0
        all_yvalidnp_drawdw = np.array(all_yvalidnp_drawdw.tolist())
        all_yvalidnp_drawdw[:, :][np.isnan(all_yvalidnp_drawdw[:, :])] = 0
        all_yvalidnp_drawdw[:, :][np.isinf(all_yvalidnp_drawdw[:, :])] = 0
        return all_xtrainnp, all_ytrainnp_ret_a, all_ytrainnp_ret_h, all_ytrainnp_ret_l, \
               all_ytrainnp_stdup, all_ytrainnp_stddw, all_ytrainnp_drawup, all_ytrainnp_drawdw, \
               all_xvalidnp, all_yvalidnp_ret_a, all_yvalidnp_ret_h, all_yvalidnp_ret_l, \
               all_yvalidnp_stdup, all_yvalidnp_stddw, all_yvalidnp_drawup, all_yvalidnp_drawdw

    def _prepare_newtrain_data(self, train_bars, ave_list, bband_list, data_range, split=0.8):
        mult_charact_trainx = []
        mult_charact_trainy_base = []
        mult_charact_trainy_much = []
        mult_charact_validx = []
        mult_charact_validy_base = []
        mult_charact_validy_much = []
        symbol_list = list(train_bars.symbol_pre_half_std_up.keys())
        totallenth = len(train_bars.symbol_ori_data[symbol_list[0]]["close"])
        mid_lenth = max(bband_list[-1] - 1, ave_list[-1])
        trainpre_pos = max(data_range[0] if data_range[0] is not None else 0, ave_list[-1] - 1)
        validaft_lenth = max(data_range[1] if data_range[1] is not None else 0, bband_list[-1])
        usefull_lenth = totallenth - trainpre_pos - validaft_lenth - mid_lenth
        trainaft_pos = int(usefull_lenth * split) + trainpre_pos
        validpre_pos = trainpre_pos + int(usefull_lenth * split) + mid_lenth
        validaft_pos = trainpre_pos + usefull_lenth + mid_lenth
        print("total length: {} train range:{}-{}. valid range:{}-{}.".format(totallenth, trainpre_pos, trainaft_pos,
                                                                              validpre_pos, validaft_pos))
        for s in symbol_list:
            # 1. 加载标签数据
            print(s)
            xchara_trainlist = []
            xchara_validlist = []
            xlen_slist = len(ave_list)
            for single_chara in range(xlen_slist):
                xchara_trainlist.append(train_bars.symbol_pre_half_std_up[s][single_chara][trainpre_pos:trainaft_pos])
                xchara_trainlist.append(train_bars.symbol_pre_half_std_down[s][single_chara][trainpre_pos:trainaft_pos])
                xchara_validlist.append(train_bars.symbol_pre_half_std_up[s][single_chara][validpre_pos:validaft_pos])
                xchara_validlist.append(train_bars.symbol_pre_half_std_down[s][single_chara][validpre_pos:validaft_pos])
                for single2_chara in range(xlen_slist):
                    xchara_trainlist.append(
                        train_bars.symbol_pre_retp[s][single_chara][single2_chara][trainpre_pos:trainaft_pos])
                    xchara_trainlist.append(
                        train_bars.symbol_pre_retm[s][single_chara][single2_chara][trainpre_pos:trainaft_pos])
                    xchara_validlist.append(
                        train_bars.symbol_pre_retp[s][single_chara][single2_chara][validpre_pos:validaft_pos])
                    xchara_validlist.append(
                        train_bars.symbol_pre_retm[s][single_chara][single2_chara][validpre_pos:validaft_pos])
            tmp_xtrainnp = np.vstack(xchara_trainlist)
            tmp_xvalidnp = np.vstack(xchara_validlist)
            # aft
            ychara_base_trainlist = []
            ychara_much_trainlist = []
            ychara_base_validlist = []
            ychara_much_validlist = []
            ylen_slist = len(bband_list)
            for single_chara in range(ylen_slist):
                ychara_base_trainlist.append(train_bars.symbol_aft_reta[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_base_trainlist.append(
                    train_bars.symbol_aft_half_std_up[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_base_trainlist.append(
                    train_bars.symbol_aft_half_std_down[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_much_trainlist.append(train_bars.symbol_aft_drawup[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_much_trainlist.append(train_bars.symbol_aft_drawdown[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_much_trainlist.append(
                    train_bars.symbol_aft_retp_high[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_much_trainlist.append(train_bars.symbol_aft_retp_low[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_base_validlist.append(train_bars.symbol_aft_reta[s][single_chara][validpre_pos:validaft_pos])
                ychara_base_validlist.append(
                    train_bars.symbol_aft_half_std_up[s][single_chara][validpre_pos:validaft_pos])
                ychara_base_validlist.append(
                    train_bars.symbol_aft_half_std_down[s][single_chara][validpre_pos:validaft_pos])
                ychara_much_validlist.append(train_bars.symbol_aft_drawup[s][single_chara][validpre_pos:validaft_pos])
                ychara_much_validlist.append(train_bars.symbol_aft_drawdown[s][single_chara][validpre_pos:validaft_pos])
                ychara_much_validlist.append(
                    train_bars.symbol_aft_retp_high[s][single_chara][validpre_pos:validaft_pos])
                ychara_much_validlist.append(train_bars.symbol_aft_retp_low[s][single_chara][validpre_pos:validaft_pos])
            tmp_ytrainnp_base = np.vstack(ychara_base_trainlist)
            tmp_ytrainnp_much = np.vstack(ychara_much_trainlist)
            tmp_yvalidnp_base = np.vstack(ychara_base_validlist)
            tmp_yvalidnp_much = np.vstack(ychara_much_validlist)
            # # 2. 删除无效行
            # delpresig = np.isnan(xchara_list[-1])
            # # print(delpresig[~delpresig])
            # delaftsig = np.isnan(ychara_base_list[-1])
            # # print(delaftsig[~delaftsig])
            # delpreaftsig = np.logical_or(delpresig, delaftsig)
            tmp_xtrainnp = np.transpose(tmp_xtrainnp)
            # tmp_xnp = tmp_xnp[~delpreaftsig]
            tmp_ytrainnp_base = np.transpose(tmp_ytrainnp_base)
            # tmp_ynp_base = tmp_ynp_base[~delpreaftsig]
            tmp_ytrainnp_much = np.transpose(tmp_ytrainnp_much)
            # tmp_ynp_much = tmp_ynp_much[~delpreaftsig]
            tmp_xvalidnp = np.transpose(tmp_xvalidnp)
            # tmp_xnp = tmp_xnp[~delpreaftsig]
            tmp_yvalidnp_base = np.transpose(tmp_yvalidnp_base)
            # tmp_ynp_base = tmp_ynp_base[~delpreaftsig]
            tmp_yvalidnp_much = np.transpose(tmp_yvalidnp_much)
            # tmp_ynp_much = tmp_ynp_much[~delpreaftsig]
            mult_charact_trainx.append(tmp_xtrainnp)
            mult_charact_trainy_base.append(tmp_ytrainnp_base)
            mult_charact_trainy_much.append(tmp_ytrainnp_much)
            mult_charact_validx.append(tmp_xvalidnp)
            mult_charact_validy_base.append(tmp_yvalidnp_base)
            mult_charact_validy_much.append(tmp_yvalidnp_much)
        all_ytrainnp_base = np.vstack(mult_charact_trainy_base)
        all_ytrainnp_much = np.vstack(mult_charact_trainy_much)
        all_xtrainnp = np.vstack(mult_charact_trainx)
        all_yvalidnp_base = np.vstack(mult_charact_validy_base)
        all_yvalidnp_much = np.vstack(mult_charact_validy_much)
        all_xvalidnp = np.vstack(mult_charact_validx)
        # 4. 处理nan inf
        all_xtrainnp[:, :][np.isnan(all_xtrainnp[:, :])] = 0
        all_xtrainnp[:, :][np.isinf(all_xtrainnp[:, :])] = 0
        all_xvalidnp[:, :][np.isnan(all_xvalidnp[:, :])] = 0
        all_xvalidnp[:, :][np.isinf(all_xvalidnp[:, :])] = 0
        all_ytrainnp_much = np.array(all_ytrainnp_much.tolist())
        all_ytrainnp_much[:, :][np.isnan(all_ytrainnp_much[:, :])] = 0
        all_ytrainnp_much[:, :][np.isinf(all_ytrainnp_much[:, :])] = 0
        all_yvalidnp_much = np.array(all_yvalidnp_much.tolist())
        all_yvalidnp_much[:, :][np.isnan(all_yvalidnp_much[:, :])] = 0
        all_yvalidnp_much[:, :][np.isinf(all_yvalidnp_much[:, :])] = 0
        all_ytrainnp_base = np.array(all_ytrainnp_base.tolist())
        all_ytrainnp_base[:, :][np.isnan(all_ytrainnp_base[:, :])] = 0
        all_ytrainnp_base[:, :][np.isinf(all_ytrainnp_base[:, :])] = 0
        all_yvalidnp_base = np.array(all_yvalidnp_base.tolist())
        all_yvalidnp_base[:, :][np.isnan(all_yvalidnp_base[:, :])] = 0
        all_yvalidnp_base[:, :][np.isinf(all_yvalidnp_base[:, :])] = 0
        # print(all_xtrainnp.shape, all_ytrainnp_base.shape, all_ytrainnp_much.shape, all_xvalidnp.shape,
        #       all_yvalidnp_base.shape, all_yvalidnp_much.shape)
        return all_xtrainnp, all_ytrainnp_base, all_ytrainnp_much, all_xvalidnp, all_yvalidnp_base, all_yvalidnp_much

    def _prepare_predict_data(self, predict_bars, ave_list, data_range):
        mult_charactx = []
        symbol_list = list(predict_bars.symbol_pre_half_std_up.keys())
        for s in symbol_list:
            # 1. 加载标签数据
            xchara_list = []
            xlen_slist = len(ave_list)
            for single_chara in range(xlen_slist):
                xchara_list.append(
                    predict_bars.symbol_pre_half_std_up[s][single_chara].values[data_range[0] - 1:data_range[1]])
                xchara_list.append(
                    predict_bars.symbol_pre_half_std_down[s][single_chara].values[data_range[0] - 1:data_range[1]])
                for single2_chara in range(xlen_slist):
                    xchara_list.append(
                        predict_bars.symbol_pre_retp[s][single_chara][single2_chara].values[
                        data_range[0] - 1:data_range[1]])
                    xchara_list.append(
                        predict_bars.symbol_pre_retm[s][single_chara][single2_chara].values[
                        data_range[0] - 1:data_range[1]])
            # 2. 删除无效行
            tmp_xnp = np.vstack(xchara_list)
            tmp_xnp = np.transpose(tmp_xnp)
            mult_charactx.append(tmp_xnp)
        all_xnp = np.vstack(mult_charactx)
        # 3. 处理nan inf
        all_xnp[:, :][np.isnan(all_xnp[:, :])] = 0
        all_xnp[:, :][np.isinf(all_xnp[:, :])] = 0
        return all_xnp

    def _prepare_every_predict_data(self, predict_bars, symbol, ave_list, data_range):
        # 1. 加载标签数据
        xchara_list = []
        xlen_slist = len(ave_list)
        for single_chara in range(xlen_slist):
            xchara_list.append(
                predict_bars.symbol_pre_half_std_up[symbol][single_chara].values[data_range[0]:data_range[1]])
            xchara_list.append(
                predict_bars.symbol_pre_half_std_down[symbol][single_chara].values[data_range[0]:data_range[1]])
            for single2_chara in range(xlen_slist):
                xchara_list.append(predict_bars.symbol_pre_retp[symbol][single_chara][single2_chara].values[
                                   data_range[0]:data_range[1]])
                xchara_list.append(predict_bars.symbol_pre_retm[symbol][single_chara][single2_chara].values[
                                   data_range[0]:data_range[1]])
        all_xnp = np.vstack(xchara_list)
        all_xnp = np.transpose(all_xnp)
        # 3. 处理nan inf
        all_xnp[:, :][np.isnan(all_xnp[:, :])] = 0
        all_xnp[:, :][np.isinf(all_xnp[:, :])] = 0
        return all_xnp

    def _prepare_fake_pred_data(self, one_fake_data, ave_list):
        # 1. 加载标签数据
        xchara_list = []
        xlen_slist = len(ave_list)
        for single_chara in range(xlen_slist):
            xchara_list.append(one_fake_data["pre_half_std_up"][single_chara])
            xchara_list.append(one_fake_data["pre_half_std_down"][single_chara])
            for single2_chara in range(xlen_slist):
                xchara_list.append(one_fake_data["pre_retp"][single_chara][single2_chara])
                xchara_list.append(one_fake_data["pre_retm"][single_chara][single2_chara])
                # 2. 删除无效行
        all_xnp = np.vstack(xchara_list)
        all_xnp = np.transpose(all_xnp)
        # 3. 处理nan inf
        all_xnp[:, :][np.isnan(all_xnp[:, :])] = 0
        all_xnp[:, :][np.isinf(all_xnp[:, :])] = 0
        return all_xnp

    def train_probability_everysignals(self, train_bars, ave_list, bband_list, date_range, newdata=1, split=0.8,
                                       model_paras=None, args=None):
        """训练"""
        # 1. 输入参数
        self._prepare_model_para(model_paras=model_paras, args=args)
        # 2. 生产数据 随机打乱，分成batch
        data_buff_dir = "everynpy_" + "_".join([str(i1) for i1 in bband_list])
        full_data_buff_dir = os.path.join(data_path, data_buff_dir)
        makesurepath(full_data_buff_dir)
        if os.path.isfile(os.path.join(full_data_buff_dir, "drawdw_v.npy")) and 1 != newdata:
            print("loading old data")
            inputs_t = np.load(os.path.join(full_data_buff_dir, "inputs_t.npy"))
            reta_t = np.load(os.path.join(full_data_buff_dir, "reta_t.npy"))
            reth_t = np.load(os.path.join(full_data_buff_dir, "reth_t.npy"))
            retl_t = np.load(os.path.join(full_data_buff_dir, "retl_t.npy"))
            stdup_t = np.load(os.path.join(full_data_buff_dir, "stdup_t.npy"))
            stddw_t = np.load(os.path.join(full_data_buff_dir, "stddw_t.npy"))
            drawup_t = np.load(os.path.join(full_data_buff_dir, "drawup_t.npy"))
            drawdw_t = np.load(os.path.join(full_data_buff_dir, "drawdw_t.npy"))
            inputs_v = np.load(os.path.join(full_data_buff_dir, "inputs_v.npy"))
            reta_v = np.load(os.path.join(full_data_buff_dir, "reta_v.npy"))
            reth_v = np.load(os.path.join(full_data_buff_dir, "reth_v.npy"))
            retl_v = np.load(os.path.join(full_data_buff_dir, "retl_v.npy"))
            stdup_v = np.load(os.path.join(full_data_buff_dir, "stdup_v.npy"))
            stddw_v = np.load(os.path.join(full_data_buff_dir, "stddw_v.npy"))
            drawup_v = np.load(os.path.join(full_data_buff_dir, "drawup_v.npy"))
            drawdw_v = np.load(os.path.join(full_data_buff_dir, "drawdw_v.npy"))
            # 打印尺寸
            print_range(train_bars, bband_list, ave_list, date_range, split)
        else:
            # 2. 加载衍生前值
            print("gene new data")
            train_bars.generate_b_derivative()
            # 2. 加载衍生后值
            train_bars.generate_a_derivative()
            inputs_t, reta_t, reth_t, retl_t, stdup_t, stddw_t, drawup_t, drawdw_t, inputs_v, reta_v, reth_v, retl_v, \
            stdup_v, stddw_v, drawup_v, drawdw_v = self._prepare_every_train_data(train_bars, ave_list, bband_list,
                                                                                  date_range, split)
            np.save(os.path.join(full_data_buff_dir, "inputs_t"), inputs_t)
            np.save(os.path.join(full_data_buff_dir, "reta_t"), reta_t)
            np.save(os.path.join(full_data_buff_dir, "reth_t"), reth_t)
            np.save(os.path.join(full_data_buff_dir, "retl_t"), retl_t)
            np.save(os.path.join(full_data_buff_dir, "stdup_t"), stdup_t)
            np.save(os.path.join(full_data_buff_dir, "stddw_t"), stddw_t)
            np.save(os.path.join(full_data_buff_dir, "drawup_t"), drawup_t)
            np.save(os.path.join(full_data_buff_dir, "drawdw_t"), drawdw_t)
            np.save(os.path.join(full_data_buff_dir, "inputs_v"), inputs_v)
            np.save(os.path.join(full_data_buff_dir, "reta_v"), reta_v)
            np.save(os.path.join(full_data_buff_dir, "reth_v"), reth_v)
            np.save(os.path.join(full_data_buff_dir, "retl_v"), retl_v)
            np.save(os.path.join(full_data_buff_dir, "stdup_v"), stdup_v)
            np.save(os.path.join(full_data_buff_dir, "stddw_v"), stddw_v)
            np.save(os.path.join(full_data_buff_dir, "drawup_v"), drawup_v)
            np.save(os.path.join(full_data_buff_dir, "drawdw_v"), drawdw_v)
        # 3. 训练
        print(inputs_t.shape, reta_t.shape, reth_t.shape, retl_t.shape,
              stdup_t.shape, stddw_t.shape, drawup_t.shape, drawdw_t.shape,
              inputs_v.shape, reta_v.shape, reth_v.shape, retl_v.shape,
              stdup_v.shape, stddw_v.shape, drawup_v.shape, drawdw_v.shape)
        print("start-training")
        self.trainconfig["tailname"] += data_buff_dir
        # self.trainconfig["inputdim"] = inputs_t.shape[1]
        # self.trainconfig["outretdim"], self.trainconfig["outstddim"] = targets_base_t.shape[1], targets_much_t.shape[1]
        modelcrnn = CRNNevery(ave_list, bband_list, config=self.trainconfig)
        modelcrnn.buildModel()
        batch_size = self.trainconfig["batchsize"]
        num_epochs = self.trainconfig["epoch"]
        globalstep = modelcrnn.batch_train(inputs_t, reta_t, reth_t, retl_t, stdup_t, stddw_t, drawup_t, drawdw_t,
                                           inputs_v, reta_v, reth_v, retl_v, stdup_v, stddw_v, drawup_v, drawdw_v,
                                           batch_size, num_epochs)

    def predict_probability_signals(self, predict_bars, ave_list, bband_list, date_range, model_paras=None, args=None):
        """预测"""
        # 1. 输入参数
        self._prepare_model_para(model_paras=model_paras, args=args)
        # 2. 生产数据
        self.trainconfig["dropout"] = 1.0
        data_buff_dir = "everynpy_" + "_".join([str(i1) for i1 in bband_list])
        self.trainconfig["tailname"] += data_buff_dir
        modelcrnn = CRNNevery(ave_list, bband_list, config=self.trainconfig)
        modelcrnn.buildModel()
        # 3. 预测结果
        pred_list_json = {}
        for symbol in predict_bars.symbol_list:
            inputs_t = self._prepare_every_predict_data(predict_bars, symbol, ave_list, date_range)
            # print("inputs_t")
            # print(len(inputs_t))
            # print(len(inputs_t[0]))
            # print(inputs_t)
            pred_list_json[symbol] = modelcrnn.predict(inputs_t)
        return pred_list_json

    def predict_fake_proba_signals(self, predict_bars, ave_list, bband_list, showconfig, model_paras=None, args=None):
        """预测"""
        # 1. 输入参数
        self._prepare_model_para(model_paras=model_paras, args=args)
        # 2. 生产数据
        self.trainconfig["dropout"] = 1.0
        data_buff_dir = "everynpy_" + "_".join([str(i1) for i1 in bband_list])
        self.trainconfig["tailname"] += data_buff_dir
        modelcrnn = CRNNevery(ave_list, bband_list, config=self.trainconfig)
        modelcrnn.buildModel()
        # 3. 预测结果
        pred_list_json = {}
        print("generate_lastspace: ")
        fake_data, fake_ori = predict_bars.generate_lastspace(**showconfig)
        for symbol in predict_bars.symbol_list:
            print("fake: ", symbol)
            inputs_t = self._prepare_fake_pred_data(fake_data[symbol], ave_list)
            pred_list_json[symbol] = modelcrnn.predict(inputs_t)
        return pred_list_json, fake_ori
