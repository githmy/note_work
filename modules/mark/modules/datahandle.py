from __future__ import print_function
import os
import numpy as np
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
from modules.event import *
from modules.stocks.finance_tool import ElementTool
from modules.stocks.stock_data2 import TSstockScrap
from utils.log_tool import *

'''
DataHandler是一个抽象数据处理类，所以实际数据处理类都继承于此（包含历史回测、实盘）
'''


class DataHandler(object):
    __metaclass__ = ABCMeta

    # 返回最近的数据条目
    @abstractmethod
    def get_latest_bar(self, symbol):
        raise NotImplementedError("没有实现 get_latest_bar()")

    # 返回最近N条数据条目
    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        返回最近的N个Bar,如果当前Bar的数量不足N，则有多少就返回多少
        """
        raise NotImplementedError("没有实现 get_latest_bars()")

    # 返回数据条目的Python datetime object
    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        raise NotImplementedError("没有实现 get_latest_bar_datetime()")

    # 返回数据条目的Open,High,Low,Close,Volume,OI数据
    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        raise NotImplementedError("没有实现 get_latest_bar_value()")

    # 返回N条数据，没有则返回N-k条数据
    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        raise NotImplementedError("没有实现 get_latest_bars_values()")

    # 数据条目放到序列中
    @abstractmethod
    def update_bars(self):
        """
        把symbol列表里所有symbol最近的Bar导入
        """
        raise NotImplementedError("没有实现 update_bars()")


class CSVDataHandler(DataHandler):
    def __init__(self, events, csv_dir, symbol_list, ave_list, mount_list):
        self.events = events
        # symbol_list:传入要处理的symbol列表集合，list类型
        self.symbol_list = symbol_list
        self.symbol_list_with_benchmark = copy.deepcopy(self.symbol_list)

        self.csv_dir = csv_dir
        self.ave_list = ave_list
        self.mount_list = mount_list

        self.symbol_ori_data = {}  # symbol_data，{symbol:DataFrame}
        self.latest_symbol_data = {}  # 最新的bar + 累计的旧值:{symbol:[bar1,bar2,barNew]}
        self.b_continue_backtest = True
        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        comb_index = None
        for s in self.symbol_list_with_benchmark:
            # 加载csv文件,date,OHLC,Volume
            self.symbol_ori_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s), header=0, index_col=0, parse_dates=False,
                names=['date', 'open', 'high', 'low', 'close', 'volume']).sort_index()
            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_ori_data[s].index
            else:
                # 这里要赋值，否则comb_index还是原来的index
                comb_index = comb_index.union(self.symbol_ori_data[s].index)
            # 设置latest symbol_data 为 None
            self.latest_symbol_data[s] = []
        # Reindex the dataframes
        for s in self.symbol_list_with_benchmark:
            # pad方式，就是用前一天的数据再填充这一天的丢失，对于资本市场这是合理的，比如这段时间停牌。那就是按停牌前一天的价格数据来计算。
            self.symbol_ori_data[s] = self.symbol_ori_data[s].reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        row = (index,series),row[0]=index,row[1]=[OHLCV]
        """
        for row in self.symbol_ori_data[symbol]:
            row_dict = {'symbol': symbol, 'date': row[1][0], 'open': row[1][1], 'high': row[1][2], 'low': row[1][3],
                        'close': row[1][4], 'volume': row[1][5]}
            yield row_dict

    def update_bars(self):
        """
        循环每只股票的下一个值
        """
        for s in self.symbol_list_with_benchmark:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.b_continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(BarEvent())

    def get_latest_bar(self, symbol):
        """
        返回最近的 N bars 或 不够时 N-k 个.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        返回最近的 N bars 或 不够时 N-k 个.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        return bars_list[-1]["date"]

    def get_latest_bar_value(self, symbol, val_type):
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        # return getattr(bars_list[-1][1], val_type)
        return bars_list[-1][val_type]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        返回最近的 N bars 或 不够时 N-k 个 某类型值.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        return np.array([b[val_type] for b in bars_list])
        # return np.array([getattr(b, val_type) for b in bars_list])


class CSVAppendDataHandler(CSVDataHandler):
    def _open_convert_csv_files(self):
        comb_index = None
        for s in self.symbol_list_with_benchmark:
            # 加载csv文件,date,OHLC,Volume
            self.symbol_ori_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s), header=0, index_col=0, parse_dates=False,
                names=['date', 'open', 'high', 'low', 'close', 'volume']).sort_index()
            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_ori_data[s].index
            else:
                # 这里要赋值，否则comb_index还是原来的index
                comb_index = comb_index.union(self.symbol_ori_data[s].index)
            # 设置latest symbol_data 为 None
            self.latest_symbol_data[s] = []
        # Reindex the dataframes
        for s in self.symbol_list_with_benchmark:
            # 这是一个发生器iterrows[index,series],用next(self.symbol_ori_data[s])
            # pad方式，就是用前一天的数据再填充这一天的丢失，对于资本市场这是合理的，比如这段时间停牌。那就是按停牌前一天的价格数据来计算。
            self.symbol_ori_data[s] = self.symbol_ori_data[s].reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        row = (index,series),row[0]=index,row[1]=[OHLCV]
        """
        for row in self.symbol_ori_data[symbol]:
            row_dict = {'symbol': symbol, 'date': row[1][0], 'open': row[1][1], 'high': row[1][2], 'low': row[1][3],
                        'close': row[1][4], 'volume': row[1][5]}
            yield row_dict

    def update_bars(self):
        """
        循环每只股票的下一个值
        """
        for s in self.symbol_list_with_benchmark:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.b_continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(BarEvent())

    def get_latest_bar(self, symbol):
        """
        返回最近的 N bars 或 不够时 N-k 个.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        返回最近的 N bars 或 不够时 N-k 个.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        return bars_list[-1]["date"]

    def get_latest_bar_value(self, symbol, val_type):
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        # return getattr(bars_list[-1][1], val_type)
        return bars_list[-1][val_type]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        返回最近的 N bars 或 不够时 N-k 个 某类型值.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        return np.array([b[val_type] for b in bars_list])
        # return np.array([getattr(b, val_type) for b in bars_list])


# 直接加载满
class LoadCSVHandler(object):
    def __init__(self, events, csv_dir, symbol_list, ave_list, bband_list):
        self.events = events
        # symbol_list:传入要处理的symbol列表集合，list类型
        self.symbol_list = symbol_list
        self.symbol_list_with_benchmark = copy.deepcopy(self.symbol_list)

        self.csv_dir = csv_dir
        self.ave_list = ave_list
        self.bband_list = bband_list

        self.symbol_ori_data = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_avep = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_avem = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_half_std_up = {}
        self.symbol_pre_half_std_down = {}
        self.symbol_pre_retp = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_retm = {}  # symbol_data，{symbol:DataFrame}

        # self.symbol_aft_retp = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_aft_retp_high = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_aft_retp_low = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_aft_reta = {}
        self.symbol_aft_half_std_up = {}
        self.symbol_aft_half_std_down = {}
        self.symbol_aft_drawup = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_aft_drawdown = {}  # symbol_data，{symbol:DataFrame}
        # 工具类实例化
        self.tool_ins = ElementTool()
        # 加载原始值
        if self.csv_dir is None:
            self._get_net_csv2files()
        else:
            self._open_convert_csv_files()

    def _get_net_csv2files(self):
        dclass = TSstockScrap(data_path)
        startdate = "2000-01-01 00:00:00"
        dclass.scrap_all_n_store(startdate)

    def _open_convert_csv_files(self):
        comb_index = None
        for s in self.symbol_list_with_benchmark:
            # # 加载csv文件,date,OHLC,Volume
            # self.symbol_ori_data[s] = pd.read_csv(
            #     os.path.join(self.csv_dir, '%s.csv' % s), header=0, index_col=0, parse_dates=False,
            #     names=['date', 'open', 'high', 'low', 'close', 'volume']).sort_index()
            self.symbol_ori_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s), header=0, index_col=None, parse_dates=False).sort_index()
            self.symbol_ori_data[s] = self.symbol_ori_data[s][['date', 'open', 'high', 'low', 'close', 'volume']]
            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_ori_data[s].index
            else:
                # 这里要赋值，否则comb_index还是原来的index
                comb_index = comb_index.union(self.symbol_ori_data[s].index)
                # 设置latest symbol_data 为 None
        # Reindex the dataframes
        for s in self.symbol_list_with_benchmark:
            # pad方式，就是用前一天的数据再填充这一天的丢失，对于资本市场这是合理的，比如这段时间停牌。那就是按停牌前一天的价格数据来计算。
            self.symbol_ori_data[s] = self.symbol_ori_data[s].reindex(index=comb_index, method='pad')

    # 加载衍生前值
    def generate_b_derivative(self):
        for s in self.symbol_list_with_benchmark:
            self.symbol_pre_avep[s] = []
            self.symbol_pre_avem[s] = []
            self.symbol_pre_half_std_up[s] = []
            self.symbol_pre_half_std_down[s] = []

            # 二维数组，第一维均线 第二维 涨幅
            self.symbol_pre_retp[s] = []
            self.symbol_pre_retm[s] = []
            for aven in self.ave_list:
                self.symbol_pre_retp[s].append([])
                self.symbol_pre_retm[s].append([])
                # 临时均线数据
                self.symbol_pre_avep[s].append(self.tool_ins.smaCal(self.symbol_ori_data[s]["close"], aven))
                self.symbol_pre_avem[s].append(self.tool_ins.smaCal(self.symbol_ori_data[s]["volume"], aven))
                # 方差
                # tmpup, tmpdown = self.tool_ins.pre_up_down_std(self.symbol_ori_data[s]["close"], aven)
                tmpup, tmpdown = self.tool_ins.general_pre_up_down_std(self.symbol_ori_data[s]["close"], aven)
                self.symbol_pre_half_std_up[s].append(tmpup)
                self.symbol_pre_half_std_down[s].append(tmpdown)
                # 待求涨幅值
                for avem in self.ave_list:
                    self.symbol_pre_retp[s][-1].append(self.tool_ins.rise_n(self.symbol_pre_avep[s][-1], avem))
                    self.symbol_pre_retm[s][-1].append(self.tool_ins.rise_n(self.symbol_pre_avem[s][-1], avem))

    # 加载衍生后值
    def generate_a_derivative(self):
        for s in self.symbol_list_with_benchmark:
            # self.symbol_aft_retp[s] = []
            self.symbol_aft_retp_high[s] = []
            self.symbol_aft_retp_low[s] = []
            self.symbol_aft_reta[s] = []
            self.symbol_aft_half_std_up[s] = []
            self.symbol_aft_half_std_down[s] = []
            self.symbol_aft_drawdown[s] = []
            self.symbol_aft_drawup[s] = []
            for aven in self.bband_list:
                # 未来n天的 最大涨跌幅
                self.symbol_aft_retp_high[s].append(
                    self.tool_ins.max_highlow_ret_aft_n(self.symbol_ori_data[s], aven)[0])
                self.symbol_aft_retp_low[s].append(
                    self.tool_ins.max_highlow_ret_aft_n(self.symbol_ori_data[s], aven)[1])
                tmpup, tmpdown = self.tool_ins.max_fallret_raiseret_aft_n(self.symbol_ori_data[s]["close"], aven)
                self.symbol_aft_drawup[s].append(tmpup)
                self.symbol_aft_drawdown[s].append(tmpdown)
                # 涨幅
                self.symbol_aft_reta[s].append(
                    self.tool_ins.rise_n(self.symbol_ori_data[s]["close"], aven).shift(-aven + 1))
                # 临时均线数据
                # 方差 未来n天的 上下半std
                tmpup, tmpdown = self.tool_ins.pre_up_down_std(self.symbol_ori_data[s]["close"], aven)
                self.symbol_aft_half_std_up[s].append(tmpup.shift(-aven + 1))
                self.symbol_aft_half_std_down[s].append(tmpdown.shift(-aven + 1))

    # 生成最后一日的空间 前一日的[-0.1~0.1]
    def generate_lastspace(self, range_low=-10, range_high=11, range_eff=0.01, mount_low=-10, mount_high=11,
                           mount_eff=0.01):
        # 1. 如果超前日期不足以生成明日的特征，raise
        # 2. 生产横轴为价位，纵轴为标的和操作比率数量
        data_obj = {}
        data_ori = {}
        symfack_pre_avep = {}
        symfack_pre_avem = {}
        symfack_pre_half_std_up = {}
        symfack_pre_half_std_down = {}
        # 二维数组，第一维均线 第二维 涨幅
        symfack_pre_retp = {}
        symfack_pre_retm = {}
        for s in self.symbol_list_with_benchmark:
            symfack_pre_avep[s] = []
            symfack_pre_avem[s] = []
            symfack_pre_half_std_up[s] = []
            symfack_pre_half_std_down[s] = []
            symfack_pre_retp[s] = []
            symfack_pre_retm[s] = []
            tmp_ori = []
            for aven in self.ave_list:
                # 临时均线数据
                tmp_pre_avep = []
                tmp_pre_avem = []
                tmp_up = []
                tmp_down = []
                tmp_ori = []
                for lastret in range(range_low, range_high):
                    for lastmount in range(mount_low, mount_high):
                        tmpclose = self.symbol_ori_data[s]["close"][-self.ave_list[-1] - 1:]
                        tmp_x = tmpclose.values[-2] * (1 + lastret * range_eff)
                        tmpclose.values[-1] = tmp_x
                        tmp_ori.append(tmp_x)
                        tmpvolume = self.symbol_ori_data[s]["volume"][-self.ave_list[-1] - 1:]
                        tmp_em = lastmount * mount_eff if lastmount < 0 else lastmount
                        tmpvolume.values[-1] = (1 + tmp_em) * tmpvolume.values[-2]
                        tmp_pre_avep.append(self.tool_ins.smaCal(tmpclose, aven))
                        tmp_pre_avem.append(self.tool_ins.smaCal(tmpvolume, aven))
                        tmpup, tmpdown = self.tool_ins.general_pre_up_down_std(tmpclose, aven)
                        tmp_up.append(tmpup.values[-1])
                        tmp_down.append(tmpdown.values[-1])
                # 每个均线 多个分组
                symfack_pre_avep[s].append(tmp_pre_avep)
                symfack_pre_avem[s].append(tmp_pre_avem)
                # 方差
                symfack_pre_half_std_up[s].append(np.array(tmp_up))
                symfack_pre_half_std_down[s].append(np.array(tmp_down))
                # 待求涨幅值
                symfack_pre_retp[s].append([])
                symfack_pre_retm[s].append([])
                for avem in self.ave_list:
                    tmp_pre_retp = []
                    tmp_pre_retm = []
                    lenth_p = len(range(range_low, range_high))
                    lenth_m = len(range(mount_low, mount_high))
                    for idm in range(lenth_p * lenth_m):
                        tmp_pre_retp.append(self.tool_ins.rise_n(symfack_pre_avep[s][-1][idm], avem).values[-1])
                        tmp_pre_retm.append(self.tool_ins.rise_n(symfack_pre_avem[s][-1][idm], avem).values[-1])
                    symfack_pre_retp[s][-1].append(np.array(tmp_pre_retp))
                    symfack_pre_retm[s][-1].append(np.array(tmp_pre_retm))
            data_obj[s] = {
                "pre_half_std_up": symfack_pre_half_std_up[s],
                "pre_half_std_down": symfack_pre_half_std_down[s],
                "pre_retp": symfack_pre_retp[s],
                "pre_retm": symfack_pre_retm[s],
            }
            data_ori[s] = np.array(tmp_ori)
        return data_obj, data_ori
