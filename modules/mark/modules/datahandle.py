from __future__ import print_function
import os
import numpy as np
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
from utils.path_tool import makesurepath
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
        self.open_convert_csv_files()

    def open_convert_csv_files(self):
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
    def open_convert_csv_files(self):
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
    def __init__(self, events, csv_dir, symbol_list, ave_list, bband_list, uband_list):
        self.events = events
        # symbol_list:传入要处理的symbol列表集合，list类型
        self.symbol_list = symbol_list
        self.symbol_list_with_benchmark = copy.deepcopy(self.symbol_list)

        self.csv_dir = csv_dir
        self.ave_list = ave_list
        self.bband_list = bband_list
        self.uband_list = uband_list
        self.symbol_ori_data = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_avep = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_avem = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_half_std_up = {}
        self.symbol_pre_half_std_down = {}
        self.symbol_pre_retp = {}  # symbol_data，{symbol:DataFrame}
        self.symbol_pre_retm = {}  # symbol_data，{symbol:DataFrame}

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
            self.open_convert_csv_files()

    def _get_net_csv2files(self):
        dclass = TSstockScrap(data_path)
        startdate = "2000-01-01 00:00:00"
        dclass.scrap_all_n_store(startdate)

    def get_some_net_csv2files(self, get_startdate="2000-01-01 00:00:00"):
        dclass = TSstockScrap(data_path)
        dclass.scrap_some_n_store(get_startdate, self.symbol_list)

    def get_some_current_net_csv2files(self):
        dclass = TSstockScrap(data_path)
        dclass.scrap_some_current_n_store(self.symbol_list)

    def open_convert_csv_files(self):
        comb_index = None
        for s in self.symbol_list_with_benchmark:
            # # 加载csv文件,date,OHLC,Volume
            filename = os.path.join(self.csv_dir, '%s.csv' % s)
            if not os.path.isfile(filename):
                dclass = TSstockScrap(data_path)
                startdate = "2000-01-01 00:00:00"
                dclass.single_n_store(s.replace("_D", ""), startdate)
            self.symbol_ori_data[s] = pd.read_csv(filename, header=0, index_col=0, parse_dates=False).sort_index()
            self.symbol_ori_data[s] = self.symbol_ori_data[s][['open', 'high', 'low', 'close', 'volume']]
            # self.symbol_ori_data[s] = self.symbol_ori_data[s][['date', 'open', 'high', 'low', 'close', 'volume']]
            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_ori_data[s].index
            else:
                # 这里要赋值，否则comb_index还是原来的index
                comb_index = comb_index.union(self.symbol_ori_data[s].index)
                # 设置latest symbol_data 为 None
        for s in self.symbol_list_with_benchmark:
            # pad方式，就是用前一天的数据再填充这一天的丢失，对于资本市场这是合理的，比如这段时间停牌。那就是按停牌前一天的价格数据来计算。
            # Reindex the dataframes
            self.symbol_ori_data[s] = self.symbol_ori_data[s].reindex(index=comb_index)
            self.symbol_ori_data[s] = self.symbol_ori_data[s].fillna(method='ffill')
            self.symbol_ori_data[s] = self.symbol_ori_data[s].fillna(method='bfill')
            self.symbol_ori_data[s].reset_index(level=0, inplace=True)

    def prepare_every_train_data(self, date_range, split):
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
        symbol_list = list(self.symbol_pre_half_std_up.keys())
        totallenth = len(self.symbol_ori_data[symbol_list[0]]["close"])
        mid_lenth = max(self.bband_list[-1] - 1, self.ave_list[-1])
        trainpre_pos = max(date_range[0] if date_range[0] is not None else 0, self.ave_list[-1] - 1)
        validaft_lenth = max(date_range[1] if date_range[1] is not None else 0, self.bband_list[-1])
        usefull_lenth = totallenth - trainpre_pos - validaft_lenth - mid_lenth
        trainaft_pos = int(usefull_lenth * split) + trainpre_pos
        validpre_pos = trainpre_pos + int(usefull_lenth * split) + mid_lenth
        validaft_pos = trainpre_pos + usefull_lenth + mid_lenth
        print("total length: {} train range:{}-{}. valid range:{}-{}.".format(totallenth, trainpre_pos, trainaft_pos,
                                                                              validpre_pos, validaft_pos))
        for s in self.symbol_list:
            # 1. 加载标签数据
            xchara_trainlist = []
            xchara_validlist = []
            xlen_slist = len(self.ave_list)
            for single_chara in range(xlen_slist):
                xchara_trainlist.append(self.symbol_pre_half_std_up[s][single_chara][trainpre_pos:trainaft_pos])
                xchara_trainlist.append(self.symbol_pre_half_std_down[s][single_chara][trainpre_pos:trainaft_pos])
                xchara_validlist.append(self.symbol_pre_half_std_up[s][single_chara][validpre_pos:validaft_pos])
                xchara_validlist.append(self.symbol_pre_half_std_down[s][single_chara][validpre_pos:validaft_pos])
                for single2_chara in range(xlen_slist):
                    xchara_trainlist.append(
                        self.symbol_pre_retp[s][single_chara][single2_chara][trainpre_pos:trainaft_pos])
                    xchara_trainlist.append(
                        self.symbol_pre_retm[s][single_chara][single2_chara][trainpre_pos:trainaft_pos])
                    xchara_validlist.append(
                        self.symbol_pre_retp[s][single_chara][single2_chara][validpre_pos:validaft_pos])
                    xchara_validlist.append(
                        self.symbol_pre_retm[s][single_chara][single2_chara][validpre_pos:validaft_pos])
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
            ylen_slist = len(self.bband_list)
            for single_chara in range(ylen_slist):
                ychara_ret_a_trainlist.append(self.symbol_aft_reta[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_ret_h_trainlist.append(
                    self.symbol_aft_retp_high[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_ret_l_trainlist.append(
                    self.symbol_aft_retp_low[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_stdup_trainlist.append(
                    self.symbol_aft_half_std_up[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_stddw_trainlist.append(
                    self.symbol_aft_half_std_down[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_drawup_trainlist.append(self.symbol_aft_drawup[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_drawdw_trainlist.append(
                    self.symbol_aft_drawdown[s][single_chara][trainpre_pos:trainaft_pos])
                ychara_ret_a_validlist.append(self.symbol_aft_reta[s][single_chara][validpre_pos:validaft_pos])
                ychara_ret_h_validlist.append(
                    self.symbol_aft_retp_high[s][single_chara][validpre_pos:validaft_pos])
                ychara_ret_l_validlist.append(
                    self.symbol_aft_retp_low[s][single_chara][validpre_pos:validaft_pos])
                ychara_stdup_validlist.append(
                    self.symbol_aft_half_std_up[s][single_chara][validpre_pos:validaft_pos])
                ychara_stddw_validlist.append(
                    self.symbol_aft_half_std_down[s][single_chara][validpre_pos:validaft_pos])
                ychara_drawup_validlist.append(self.symbol_aft_drawup[s][single_chara][validpre_pos:validaft_pos])
                ychara_drawdw_validlist.append(
                    self.symbol_aft_drawdown[s][single_chara][validpre_pos:validaft_pos])
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

    def generate_chara_file(self, date_range, split=0.8):
        print("gene new data")
        self.generate_b_derivative()
        # 2. 加载衍生后值
        self.generate_a_derivative(self.bband_list)
        data_buff_dir = "everynpy_all"
        full_data_buff_dir = os.path.join(data_path, data_buff_dir)
        makesurepath(full_data_buff_dir)
        inputs_t, reta_t, reth_t, retl_t, stdup_t, stddw_t, drawup_t, drawdw_t, inputs_v, reta_v, reth_v, retl_v, \
        stdup_v, stddw_v, drawup_v, drawdw_v = self.prepare_every_train_data(date_range, split)
        np.save(os.path.join(full_data_buff_dir, "inputs_t"), inputs_t)
        np.save(os.path.join(full_data_buff_dir, "inputs_v"), inputs_v)
        for id2, i2 in enumerate(self.bband_list):
            np.save(os.path.join(full_data_buff_dir, "reta_t_{}".format(i2)), reta_t[:, id2])
            np.save(os.path.join(full_data_buff_dir, "reth_t_{}".format(i2)), reth_t[:, id2])
            np.save(os.path.join(full_data_buff_dir, "retl_t_{}".format(i2)), retl_t[:, id2])
            np.save(os.path.join(full_data_buff_dir, "stdup_t_{}".format(i2)), stdup_t[:, id2])
            np.save(os.path.join(full_data_buff_dir, "stddw_t_{}".format(i2)), stddw_t[:, id2])
            np.save(os.path.join(full_data_buff_dir, "drawup_t_{}".format(i2)), drawup_t[:, id2])
            np.save(os.path.join(full_data_buff_dir, "drawdw_t_{}".format(i2)), drawdw_t[:, id2])
            np.save(os.path.join(full_data_buff_dir, "reta_v_{}".format(i2)), reta_v[:, id2])
            np.save(os.path.join(full_data_buff_dir, "reth_v_{}".format(i2)), reth_v[:, id2])
            np.save(os.path.join(full_data_buff_dir, "retl_v_{}".format(i2)), retl_v[:, id2])
            np.save(os.path.join(full_data_buff_dir, "stdup_v_{}".format(i2)), stdup_v[:, id2])
            np.save(os.path.join(full_data_buff_dir, "stddw_v_{}".format(i2)), stddw_v[:, id2])
            np.save(os.path.join(full_data_buff_dir, "drawup_v_{}".format(i2)), drawup_v[:, id2])
            np.save(os.path.join(full_data_buff_dir, "drawdw_v_{}".format(i2)), drawdw_v[:, id2])

    def _print_range(self, data_range, split):
        totallenth = len(self.symbol_ori_data[self.symbol_list[0]]["close"])
        mid_lenth = max(self.bband_list[-1] - 1, self.ave_list[-1])
        trainpre_pos = max(data_range[0] if data_range[0] is not None else 0, self.ave_list[-1] - 1)
        validaft_lenth = max(data_range[1] if data_range[1] is not None else 0, self.bband_list[-1])
        usefull_lenth = totallenth - trainpre_pos - validaft_lenth - mid_lenth
        trainaft_pos = int(usefull_lenth * split) + trainpre_pos
        validpre_pos = trainpre_pos + int(usefull_lenth * split) + mid_lenth
        validaft_pos = trainpre_pos + usefull_lenth + mid_lenth
        print("total length: {} train range:{}-{}. valid range:{}-{}.".format(totallenth, trainpre_pos, trainaft_pos,
                                                                              validpre_pos, validaft_pos))

    def load_chara_file(self, date_range, split=0.8):
        data_buff_dir = "everynpy_all"
        full_data_buff_dir = os.path.join(data_path, data_buff_dir)
        inputs_t = np.load(os.path.join(full_data_buff_dir, "inputs_t.npy"))
        inputs_v = np.load(os.path.join(full_data_buff_dir, "inputs_v.npy"))
        shape_inputs_t = inputs_t.shape
        shape_inputs_v = inputs_v.shape
        lenth_y = len(self.bband_list)
        reta_t = np.zeros((shape_inputs_t[0], lenth_y))
        reth_t = np.zeros((shape_inputs_t[0], lenth_y))
        retl_t = np.zeros((shape_inputs_t[0], lenth_y))
        stdup_t = np.zeros((shape_inputs_t[0], lenth_y))
        stddw_t = np.zeros((shape_inputs_t[0], lenth_y))
        drawup_t = np.zeros((shape_inputs_t[0], lenth_y))
        drawdw_t = np.zeros((shape_inputs_t[0], lenth_y))
        reta_v = np.zeros((shape_inputs_v[0], lenth_y))
        reth_v = np.zeros((shape_inputs_v[0], lenth_y))
        retl_v = np.zeros((shape_inputs_v[0], lenth_y))
        stdup_v = np.zeros((shape_inputs_v[0], lenth_y))
        stddw_v = np.zeros((shape_inputs_v[0], lenth_y))
        drawup_v = np.zeros((shape_inputs_v[0], lenth_y))
        drawdw_v = np.zeros((shape_inputs_v[0], lenth_y))
        for id2, i2 in enumerate(self.bband_list):
            reta_t[:, id2] = np.load(os.path.join(full_data_buff_dir, "reta_t_{}.npy".format(i2)))
            reth_t[:, id2] = np.load(os.path.join(full_data_buff_dir, "reth_t_{}.npy".format(i2)))
            retl_t[:, id2] = np.load(os.path.join(full_data_buff_dir, "retl_t_{}.npy".format(i2)))
            stdup_t[:, id2] = np.load(os.path.join(full_data_buff_dir, "stdup_t_{}.npy".format(i2)))
            stddw_t[:, id2] = np.load(os.path.join(full_data_buff_dir, "stddw_t_{}.npy".format(i2)))
            drawup_t[:, id2] = np.load(os.path.join(full_data_buff_dir, "drawup_t_{}.npy".format(i2)))
            drawdw_t[:, id2] = np.load(os.path.join(full_data_buff_dir, "drawdw_t_{}.npy".format(i2)))
            reta_v[:, id2] = np.load(os.path.join(full_data_buff_dir, "reta_v_{}.npy".format(i2)))
            reth_v[:, id2] = np.load(os.path.join(full_data_buff_dir, "reth_v_{}.npy".format(i2)))
            retl_v[:, id2] = np.load(os.path.join(full_data_buff_dir, "retl_v_{}.npy".format(i2)))
            stdup_v[:, id2] = np.load(os.path.join(full_data_buff_dir, "stdup_v_{}.npy".format(i2)))
            stddw_v[:, id2] = np.load(os.path.join(full_data_buff_dir, "stddw_v_{}.npy".format(i2)))
            drawup_v[:, id2] = np.load(os.path.join(full_data_buff_dir, "drawup_v_{}.npy".format(i2)))
            drawdw_v[:, id2] = np.load(os.path.join(full_data_buff_dir, "drawdw_v_{}.npy".format(i2)))
        # 打印尺寸
        self._print_range(date_range, split)
        return inputs_t, reta_t, reth_t, retl_t, stdup_t, stddw_t, drawup_t, drawdw_t, inputs_v, reta_v, reth_v, retl_v, stdup_v, stddw_v, drawup_v, drawdw_v

    # 加载衍生前值
    def generate_b_derivative(self):
        for s in self.symbol_list_with_benchmark:
            print("gene before ", s)
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
                tmpup, tmpdown = self.tool_ins.general_pre_up_down_std(self.symbol_ori_data[s]["close"], aven)
                self.symbol_pre_half_std_up[s].append(tmpup)
                self.symbol_pre_half_std_down[s].append(tmpdown)
                # 待求涨幅值
                for avem in self.ave_list:
                    self.symbol_pre_retp[s][-1].append(self.tool_ins.rise_n(self.symbol_pre_avep[s][-1], avem))
                    self.symbol_pre_retm[s][-1].append(self.tool_ins.rise_n(self.symbol_pre_avem[s][-1], avem))

    # 加载衍生后值
    def generate_a_derivative(self, uband_list):
        for s in self.symbol_list_with_benchmark:
            print("generating {}".format(s))
            self.symbol_aft_retp_high[s] = []
            self.symbol_aft_retp_low[s] = []
            self.symbol_aft_reta[s] = []
            self.symbol_aft_half_std_up[s] = []
            self.symbol_aft_half_std_down[s] = []
            self.symbol_aft_drawdown[s] = []
            self.symbol_aft_drawup[s] = []
            for id2, aven in enumerate(uband_list):
                # 未来n天的 最大涨跌幅
                self.symbol_aft_retp_high[s].append(
                    self.tool_ins.general_max_highlow_ret_aft_n(self.symbol_ori_data[s], aven)[0])
                self.symbol_aft_retp_low[s].append(
                    self.tool_ins.general_max_highlow_ret_aft_n(self.symbol_ori_data[s], aven)[1])
                tmpup, tmpdown = self.tool_ins.general_max_fallret_raiseret_aft_n(self.symbol_ori_data[s]["close"],
                                                                                  aven)
                self.symbol_aft_drawup[s].append(tmpup)
                self.symbol_aft_drawdown[s].append(tmpdown)
                # 涨幅
                self.symbol_aft_reta[s].append(
                    self.tool_ins.rise_n(self.symbol_ori_data[s]["close"], aven).shift(-aven))
                # 临时均线数据,  方差 未来n天的 上下半std
                # self.symbol_aft_half_std_up[s].append(self.symbol_pre_half_std_up[s][id2].shift(-aven))
                # self.symbol_aft_half_std_down[s].append(self.symbol_pre_half_std_down[s][id2].shift(-aven))
                # self.symbol_aft_half_std_up[s].append(self.symbol_pre_half_std_up[s][id2].shift(-aven + 1))
                # self.symbol_aft_half_std_down[s].append(self.symbol_pre_half_std_down[s][id2].shift(-aven + 1))
                tmpup, tmpdown = self.tool_ins.general_pre_up_down_std(self.symbol_ori_data[s]["close"], aven)
                self.symbol_aft_half_std_up[s].append(tmpup.shift(-aven))
                self.symbol_aft_half_std_down[s].append(tmpdown.shift(-aven))

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
        for id1, s in enumerate(self.symbol_list_with_benchmark):
            print(id1, s)
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
