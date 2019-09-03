from __future__ import print_function
import os
import numpy as np
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
import itertools
from modules.event import *
from pyalgotrade import strategy


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

    def __init__(self, bars, events, ave_list, bband_list, short_window=100, long_window=400):
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
        self.short_window = short_window
        self.long_window = long_window

        # self.bars.upprb = {}
        # self.bars.downprb = {}
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

    def calculate_probability_signals(self, event):
        """
        kelly_formula
        """
        self.bars.upprb = {}
        self.bars.downprb = {}
        self.bars.f_ratio = {}
        self.bars.gain = {}
        for s in self.bars.symbol_list:
            self.bars.upprb[s] = []
            self.bars.downprb[s] = []
            self.bars.f_ratio[s] = []
            self.bars.gain[s] = []
            for id1, aven in enumerate(self.bband_list):
                tmp_rw = self.bars.symbol_aft_retp_high[s][id1] - 1.0
                tmp_rl = 1.0 - self.bars.symbol_aft_retp_low[s][id1]
                upconst = np.exp(-(tmp_rw) ** 2 / self.bars.symbol_aft_half_std_up[s][id1] ** 2)
                downconst = np.exp(-(tmp_rl) ** 2 / self.bars.symbol_aft_half_std_down[s][id1] ** 2)
                self.bars.upprb[s].append(upconst / (upconst + downconst))
                self.bars.downprb[s].append(downconst / (upconst + downconst))
                self.bars.f_ratio[s].append(
                    self.bars.upprb[s][-1] * aven / tmp_rl - self.bars.downprb[s][-1] * aven / tmp_rw)
                self.bars.gain[s].append(
                    self.bars.upprb[s][-1] * np.log(1 + self.bars.f_ratio[s][-1] * tmp_rw / aven) +
                    self.bars.downprb[s][-1] * np.log(1 - self.bars.f_ratio[s][-1] * tmp_rl / aven))
                # print(self.bars.upprb[s][-1])
                # print(self.bars.downprb[s][-1])
                # print(self.bars.f_ratio[s][-1] * tmp_rw / aven)
                # print(self.bars.f_ratio[s][-1] * tmp_rl / aven)
                # print(self.bars.gain[s][-1])
                # exit(0)
