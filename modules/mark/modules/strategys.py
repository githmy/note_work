from __future__ import print_function
import os
import numpy as np
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
from pyalgotrade import strategy
import itertools
from modules.event import *
from modules.stocks.stock_network2 import CRNN
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
    """
    各种交叉线和信号策略
    """

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

    def _prepare_train_para(self, args=None):
        # 2. 模型参数赋值
        config = {}
        parafile = os.path.join("config", "para.json")
        argspar = parseArgs(args)
        if argspar.modelname is None:
            hpara = simplejson.load(open(parafile))
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
        print()
        print("**********************************************************")
        print("parafile:", parafile)
        print("modelname:", config["modelname"])
        print("tailname:", config["tailname"])
        print("learn_rate:", config["learn_rate"])
        print("dropout:", config["dropout"])
        print("**********************************************************")
        self.trainconfig = config
        # return config

    def _prepare_train_data(self, train_bars, ave_list, bband_list, data_range, split=0.8):
        mult_charactx = []
        mult_characty_base = []
        mult_characty_much = []
        symbol_list = list(train_bars.symbol_pre_half_std_up.keys())
        for s in symbol_list:
            # 1. 加载标签数据
            xchara_list = []
            xlen_slist = len(ave_list)
            for single_chara in range(xlen_slist):
                xchara_list.append(train_bars.symbol_pre_half_std_up[s][single_chara][data_range[0] - 1:data_range[1]])
                xchara_list.append(
                    train_bars.symbol_pre_half_std_down[s][single_chara][data_range[0] - 1:data_range[1]])
                for single2_chara in range(xlen_slist):
                    xchara_list.append(
                        train_bars.symbol_pre_retp[s][single_chara][single2_chara][data_range[0] - 1:data_range[1]])
                    xchara_list.append(
                        train_bars.symbol_pre_retm[s][single_chara][single2_chara][data_range[0] - 1:data_range[1]])
            tmp_xnp = np.vstack(xchara_list)
            ychara_base_list = []
            ychara_much_list = []
            ylen_slist = len(bband_list)
            for single_chara in range(ylen_slist):
                ychara_base_list.append(train_bars.symbol_aft_reta[s][single_chara][data_range[0] - 1:data_range[1]])
                ychara_base_list.append(
                    train_bars.symbol_aft_half_std_up[s][single_chara][data_range[0] - 1:data_range[1]])
                ychara_base_list.append(
                    train_bars.symbol_aft_half_std_down[s][single_chara][data_range[0] - 1:data_range[1]])
                ychara_much_list.append(train_bars.symbol_aft_drawup[s][single_chara][data_range[0] - 1:data_range[1]])
                ychara_much_list.append(
                    train_bars.symbol_aft_drawdown[s][single_chara][data_range[0] - 1:data_range[1]])
                ychara_much_list.append(
                    train_bars.symbol_aft_retp_high[s][single_chara][data_range[0] - 1:data_range[1]])
                ychara_much_list.append(
                    train_bars.symbol_aft_retp_low[s][single_chara][data_range[0] - 1:data_range[1]])
                # for single2_chara in range(ylen_slist):
                #     ychara_ret_list.append(
                #         train_bars.symbol_aft_retp[s][single_chara][single2_chara][data_range[0] - 1:data_range[1]])
            tmp_ynp_base = np.vstack(ychara_base_list)
            tmp_ynp_much = np.vstack(ychara_much_list)
            # 2. 删除无效行
            delpresig = np.isnan(xchara_list[-1])
            # print(delpresig[~delpresig])
            delaftsig = np.isnan(ychara_base_list[-1])
            # print(delaftsig[~delaftsig])
            delpreaftsig = np.logical_or(delpresig, delaftsig)
            tmp_xnp = np.transpose(tmp_xnp)
            tmp_xnp = tmp_xnp[~delpreaftsig]
            tmp_ynp_base = np.transpose(tmp_ynp_base)
            tmp_ynp_base = tmp_ynp_base[~delpreaftsig]
            tmp_ynp_much = np.transpose(tmp_ynp_much)
            tmp_ynp_much = tmp_ynp_much[~delpreaftsig]
            mult_charactx.append(tmp_xnp)
            mult_characty_base.append(tmp_ynp_base)
            mult_characty_much.append(tmp_ynp_much)
        all_ynp_base = np.vstack(mult_characty_base)
        all_ynp_much = np.vstack(mult_characty_much)
        all_xnp = np.vstack(mult_charactx)
        # 3. split
        lenth = all_xnp.shape[0]
        mult_trainx = all_xnp[0:int(lenth * split)]
        mult_trainy_base = all_ynp_base[0:int(lenth * split)]
        mult_trainy_much = all_ynp_much[0:int(lenth * split)]
        mult_validx = all_xnp[int(lenth * split):]
        mult_validy_base = all_ynp_base[int(lenth * split):]
        mult_validy_much = all_ynp_much[int(lenth * split):]
        # 4. 处理nan inf
        mult_trainx[:, :][np.isnan(mult_trainx[:, :])] = 0
        mult_trainx[:, :][np.isinf(mult_trainx[:, :])] = 0
        mult_trainy_much[:, :][np.isnan(mult_trainy_much[:, :])] = 0
        mult_trainy_much[:, :][np.isinf(mult_trainy_much[:, :])] = 0
        mult_trainy_base[:, :][np.isnan(mult_trainy_base[:, :])] = 0
        mult_trainy_base[:, :][np.isinf(mult_trainy_base[:, :])] = 0
        mult_validx[:, :][np.isnan(mult_validx[:, :])] = 0
        mult_validx[:, :][np.isinf(mult_validx[:, :])] = 0
        mult_validy_much[:, :][np.isnan(mult_validy_much[:, :])] = 0
        mult_validy_much[:, :][np.isinf(mult_validy_much[:, :])] = 0
        mult_validy_base[:, :][np.isnan(mult_validy_base[:, :])] = 0
        mult_validy_base[:, :][np.isinf(mult_validy_base[:, :])] = 0
        return mult_trainx, mult_trainy_base, mult_trainy_much, mult_validx, mult_validy_base, mult_validy_much

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

    def train_probability_signals(self, train_bars, ave_list, bband_list, date_range, split=0.8, args=None):
        """
        训练
        """
        # 1. 输入参数
        self._prepare_train_para(args)
        # self.symbol_pre_half_std_up
        # self.symbol_pre_half_std_down
        # self.symbol_pre_retp
        # self.symbol_pre_retm

        # self.symbol_aft_reta
        # self.symbol_aft_half_std_up
        # self.symbol_aft_half_std_down

        # self.symbol_aft_drawup
        # self.symbol_aft_drawdown
        # self.symbol_aft_retp_high
        # self.symbol_aft_retp_low
        # 2. 生产数据 随机打乱，分成batch
        inputs_t, targets_base_t, targets_much_t, inputs_v, targets_base_v, targets_much_v = self._prepare_train_data(
            train_bars, ave_list, bband_list, date_range, split)
        # 3. 训练
        print(targets_base_t, targets_much_t)
        self.trainconfig["inputdim"] = inputs_t.shape[1]
        self.trainconfig["outretdim"], self.trainconfig["outstddim"] = targets_base_t.shape[1], targets_much_t.shape[1]
        modelcrnn = CRNN(ave_list, bband_list, config=self.trainconfig)
        modelcrnn.buildModel()
        batch_size, num_epochs = 32, 100000
        globalstep = modelcrnn.batch_train(inputs_t, targets_base_t, targets_much_t, inputs_v, targets_base_v,
                                           targets_much_v, batch_size, num_epochs)

    def predict_probability_signals(self, predict_bars_json, ave_list, bband_list, date_range, args=None):
        """预测"""
        # 1. 输入参数
        self._prepare_train_para(args)
        # 2. 生产数据
        self.trainconfig["dropout"] = 1.0
        modelcrnn = CRNN(ave_list, bband_list, config=self.trainconfig)
        modelcrnn.buildModel()
        # 3. 预测结果
        pred_list_json = {}
        for symbol in predict_bars_json:
            inputs_t = self._prepare_predict_data(predict_bars_json[symbol], ave_list, date_range)
            pred_list_json[symbol] = modelcrnn.predict(inputs_t)
        return pred_list_json

    def predict_fake_proba_signals(self, predict_bars, ave_list, bband_list, showconfig, args=None):
        """预测"""
        # 1. 输入参数
        self._prepare_train_para(args)
        # 2. 生产数据
        self.trainconfig["dropout"] = 1.0
        modelcrnn = CRNN(ave_list, bband_list, config=self.trainconfig)
        modelcrnn.buildModel()
        # 3. 预测结果
        pred_list_json = {}
        fake_data = predict_bars.generate_lastspace(**showconfig)
        # fake_data = predict_bars.generate_lastspace(range_low=-10, range_high=11, range_eff=0.01)
        for symbol in self.symbol_list:
            inputs_t = self._prepare_fake_pred_data(fake_data[symbol], ave_list)
            pred_list_json[symbol] = modelcrnn.predict(inputs_t)
        return pred_list_json
