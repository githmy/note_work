from __future__ import print_function

import os
import pandas as pd
import numpy as np
from utils.log_tool import *
from modules.event import FillEvent, OrderEvent
from modules.performance import create_sharpe_ratio, create_drawdowns
from modules.stocks.finance_tool import TradeTool
import math
import heapq
import copy


class Portfolio(object):
    """
    资产组合类，处理持仓和市场价值。postion DataFrame存储用时间做索引的持仓数量；
    holdings DataFrame存储特定时间索引对应代码的现金和总市场持仓价值，以及资产组合总量的百分比变化
    """

    def __init__(self, bars, events, start_date, ave_list, bband_list, initial_capital=100000.0):
        """
        根据价位和事件 起始时间 资产额， 初始化 资产组合 
        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.ave_list = ave_list
        self.bband_list = bband_list
        self.initial_capital = initial_capital
        # 初始化所有 标的 量
        self.all_positions = self.construct_all_positions()
        self.current_positions = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        # 初始化所有 标的 价 和现金 持有状态
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    # 使用开始时间字段确定时间索引开始时间，构建持仓列表
    def construct_all_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    # 使用开始时间字段确定时间索引开始时间，构建持仓列表。保存所有资产组合的开始时间的价值。
    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    # 保存所有资产组合的当前价值
    def construct_current_holdings(self):
        """
        保存所有资产组合的当前价值
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    # 在持仓矩阵中添加一条新的记录，反映了当前持仓的价值
    def update_timeindex(self, event):
        """
        在持仓矩阵中添加一条新的记录，反映了当前持仓的价值, 使用队列事件中的市场事件
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])
        dp = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dp['datetime'] = latest_datetime
        print(latest_datetime)
        for s in self.symbol_list:
            dp[s] = self.current_positions[s]
        # Append the current positions
        self.all_positions.append(dp)

        dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']
        for s in self.symbol_list:
            # market_value = self.current_positions[s] * self.bars.get_latest_bar_value(s, "adj_close")
            market_value = self.current_positions[s] * self.bars.get_latest_bar_value(s, "close")
            dh[s] = market_value
            dh['total'] += market_value
        # Append the current holdings
        self.all_holdings.append(dh)

    # 获取Fill object 更新持仓矩阵
    def update_positions_from_fill(self, fill):
        """
        Parameters:
        fill - The Fill event object to update the positions with.
        """
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1
        self.current_positions[fill.symbol] += fill_dir * fill.quantity

    # 获取Fill object 更新持仓矩阵并反映持仓市值
    def update_holdings_from_fill(self, fill):
        """
        获取Fill object 更新持仓矩阵并反映持仓市值
        Parameters:
        fill - The Fill object to update the holdings with.
        """
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1
        # fill_cost = self.bars.get_latest_bar_value(fill.symbol, "adj_close")
        fill_cost = self.bars.get_latest_bar_value(fill.symbol, "close")
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= (cost + fill.commission)

    # 收到FillEvent,更新投资组合当前持仓和市值
    def update_fill(self, event):
        """
        收到FillEvent,更新投资组合当前持仓和市值
        """
        if event.type == 'FILL':
            # 更新数量
            self.update_positions_from_fill(event)
            # 更新市值
            self.update_holdings_from_fill(event)

    # 生成一个订单对象，没有风险管理和头寸考虑
    def generate_naive_order(self, signal):
        """
        生成一个订单对象，没有风险管理和头寸考虑
        Parameters:
        signal - The tuple containing Signal information.
        """
        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength

        mkt_quantity = 100
        cur_quantity = self.current_positions[symbol]
        order_type = 'MKT'

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
        return order

    # 根据 组合逻辑 SignalEvent生成新的订单，
    def update_signals(self, event):
        """
        根据 组合逻辑 SignalEvent  生成 order_event。
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)

    # 从all_holdings创建一个pandas DataFrame
    def create_equity_curve_dataframe(self):
        """
        从all_holdings创建一个pandas DataFrame.  格式转化
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve

    # 创建资产组合总结统计列表
    def output_summary_stats(self):
        """
        创建资产组合总结统计列表。
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        sharpe_ratio = create_sharpe_ratio(returns, periods=252 * 60 * 6.5)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown
        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]
        self.equity_curve.to_csv(os.path.join(out_path, 'equity.csv'))
        return stats

    # 基于滞后统计结果 盈利测试
    def components_res_base_aft(self):
        # 1. 目标操作列表, 代号：均线考察日
        hand_unit = 100
        target_list = []
        datalenth = self.bars.symbol_ori_data[self.symbol_list[0]].shape[0]
        for i1 in range(1, datalenth + 1):
            max_bbandid = []
            max_list = []
            for s in self.bars.symbol_list:
                rank_list = [i2[i1] if not np.isnan(i2[i1]) else 0.0 for i2 in self.bars.gain[s]]
                tmp_vlaue = max(rank_list)
                max_bbandid.append(rank_list.index(tmp_vlaue))
                max_list.append(tmp_vlaue)
            day_max_val = max(max_list)
            if day_max_val > 0.0:
                symblname = self.bars.symbol_list[max_list.index(day_max_val)]
            else:
                symblname = "没有"
            target_list.append({
                symblname: max_bbandid[max_list.index(day_max_val)],
            })
        # 2. 统计值
        self.all_holdings = []
        self.all_positions = []
        for i1 in range(1, datalenth + 1):
            d = {}
            d['datetime'] = i1
            d['cash'] = self.initial_capital
            d['commission'] = 0.0
            d['total'] = self.initial_capital
            self.all_holdings.append(d)
            v = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
            self.all_positions.append(v)
        for id1, i1 in enumerate(target_list):
            key_list = list(i1.keys())
            print("id1:", id1, i1, key_list)
            if key_list[0] not in self.symbol_list:
                print("清仓")
                if id1 == 0:
                    pass
                else:
                    self.all_holdings[id1]["cash"] = self.all_holdings[id1 - 1]["cash"]
                    for i2 in self.symbol_list:
                        if id1 == 0:
                            pass
                        elif self.all_positions[id1 - 1][i2] > 0:
                            self.all_positions[id1][i2] = 0
                            self.all_holdings[id1]["cash"] += self.all_positions[id1 - 1][i2] * hand_unit * \
                                                              self.bars.symbol_ori_data[i2]["close"][
                                                                  self.all_holdings[id1]["datetime"]]
                    self.all_holdings[id1]["total"] = self.all_holdings[id1]["cash"]
            else:
                print(self.bars.f_ratio[key_list[0]][i1[key_list[0]]][self.all_holdings[id1]["datetime"]])
                if self.bars.f_ratio[key_list[0]][i1[key_list[0]]][self.all_holdings[id1]["datetime"]] > 0:
                    print("目标仓位相同")
                    # self.all_holdings[id1]["total"] = self.all_holdings[id1]["cash"]
                    if id1 == 0:
                        pass
                    else:
                        self.all_holdings[id1]["cash"] = self.all_holdings[id1 - 1]["cash"]
                        for i2 in self.symbol_list:
                            if self.all_positions[id1 - 1][i2] > 0:
                                self.all_holdings[id1]["cash"] += self.all_positions[id1 - 1][i2] * hand_unit * \
                                                                  self.bars.symbol_ori_data[i2]["close"][
                                                                      self.all_holdings[id1]["datetime"]]
                                self.all_positions[id1][i2] = 0
                    self.all_holdings[id1]["total"] = self.all_holdings[id1]["cash"]
                    targ_captail = self.bars.f_ratio[key_list[0]][i1[key_list[0]]][self.all_holdings[id1]["datetime"]] * \
                                   self.all_holdings[id1]["total"]
                    targ_mount = targ_captail / hand_unit // self.bars.symbol_ori_data[key_list[0]]["close"][
                        self.all_holdings[id1]["datetime"]]
                    self.all_positions[id1][key_list[0]] = targ_mount
                    self.all_holdings[id1]["cash"] = self.all_holdings[id1]["total"] - targ_mount * hand_unit * \
                                                                                       self.bars.symbol_ori_data[
                                                                                           key_list[0]]["close"][
                                                                                           self.all_holdings[id1][
                                                                                               "datetime"]]
                else:
                    print("目标仓位不同")
                    if id1 == 0:
                        pass
                        # elif self.all_positions[id1 - 1][i2] != 0:
                    else:
                        self.all_holdings[id1]["cash"] = self.all_holdings[id1 - 1]["cash"]
                        for i2 in self.symbol_list:
                            self.all_positions[id1][i2] = 0
                            self.all_holdings[id1]["cash"] += self.all_positions[id1 - 1][i2] * hand_unit * \
                                                              self.bars.symbol_ori_data[i2]["close"][
                                                                  self.all_holdings[id1]["datetime"]]
                    self.all_holdings[id1]["total"] = self.all_holdings[id1]["cash"]
            print(self.all_holdings[id1], self.all_positions[id1], self.bars.symbol_ori_data["SAPower"]["close"][
                self.all_holdings[id1]["datetime"]])
            # return self.all_holdings

    def components_res_every_predict(self, predict_bars, pred_list_json, policy_config, strategy_config, date_range):
        # 1. 参数初始化
        f_ratio_json = {}
        gain_json = {}
        for symbols in predict_bars.symbol_list:
            pred_list = pred_list_json[symbols]
            # 2. 计算各种概率和收益列表
            # pred_list: y_reta, y_reth, y_retl, y_stdup, y_stddw, y_drawup, y_drawdw
            upprb, downprb, f_ratio, gain = self.calculate_probability_every_signals(predict_bars.uband_list, pred_list)
            # f_ratio 必然大于零
            f_ratio_json[symbols] = f_ratio
            gain_json[symbols] = gain
        # 3. 目标操作列表, 代号：均线考察日
        print("gain_json 已经按band换算成单日: symbols , band_n, n天")
        # print(gain_json)
        # 选操作目标
        target_idlist, target_vallist = self.get_target_every_list(predict_bars.symbol_list, gain_json, strategy_config)
        print("target_idlist: n天，列表:band_idex")
        print(target_idlist)
        # 2. 统计值
        # 2.1 只用收盘价
        # all_holdings, all_positions, all_ratios = self.simu_gains_close_history(policy_config, strategy_config,
        #                                                                         predict_bars, target_idlist,
        #                                                                         f_ratio_json, date_range)
        # 2.2 每天看情况报价
        all_holdings, all_positions, all_ratios, \
        all_oper_price = self.simu_gains_every_history(policy_config, strategy_config, predict_bars, pred_list_json,
                                                       target_idlist, gain_json, f_ratio_json, date_range)
        return all_holdings, all_positions, all_ratios, all_oper_price

    def components_res_fake_predict(self, predict_bars, pred_list_json, fake_ori, policy_config):
        # 1. 参数初始化
        gain_json = {}
        f_ratio_json = {}
        mount_json = {}
        for symbols in self.symbol_list:
            pred_list = pred_list_json[symbols]
            # 2. 计算各种概率和收益列表
            upprb, downprb, f_ratio, gain = self.calculate_probability_every_signals(predict_bars.bband_list, pred_list)
            # f_ratio 必然大于零
            f_ratio_json[symbols] = f_ratio
            gain_json[symbols] = gain
            tmp_mount = []
            for i1 in fake_ori[symbols]:
                tmp_mount.append(policy_config["initial_capital"] / policy_config["hand_unit"] // i1)
            mount_json[symbols] = np.array(tmp_mount)
        return gain_json, f_ratio_json, mount_json

    def calcu_fee(self, commission, commission_rate, stamp_in, stamp_out, mount_in, mount_out, price_c, hand_unit):
        bigcost = price_c * hand_unit * (mount_in + mount_out) * commission_rate
        basicost = commission if bigcost < commission else bigcost
        costs = basicost + (stamp_in * mount_in + stamp_out * mount_out) * price_c * hand_unit
        return costs

    def get_target_every_list(self, symbol_list, gain_json, strategy_config):
        # gain_json 已经按band换算成单日: symbols , band_n, n天
        target_list = []
        target_idlist = []
        target_vallist = []
        datalenth = len(gain_json[self.symbol_list[0]][0])
        target_use_num = min([len(symbol_list), strategy_config["oper_num"]])
        for i1 in range(0, datalenth):
            # 1. 每日筛选
            max_bbandid = []
            max_list = []
            # 2. 每个 symbol 选 最大的bband_n做 目标值
            for s in symbol_list:
                # 不同均线筛选
                rank_list = [i2[i1] if not np.isnan(i2[i1]) else 0.0 for i2 in gain_json[s]]
                tmp_vlaue = max(rank_list)
                max_bbandid.append(rank_list.index(tmp_vlaue))
                max_list.append(tmp_vlaue)
            element_indexs = np.argsort(-np.array(max_list), axis=0)  # 对a按每行中元素从小到大排序, 列表加-号，即降序
            tmpvalobj = {}
            tmpidobj = {}
            # 截断 小于等于 symbel 列表
            element_indexs = element_indexs[0:target_use_num]
            for id2, i2 in enumerate(element_indexs):
                # if id2 > target_use_num:
                #     break
                day_max_val = max_list[i2]
                if day_max_val > strategy_config["thresh_low"] and day_max_val < strategy_config["thresh_high"]:
                    symblname = symbol_list[i2]
                else:
                    symblname = "没有"
                tmpvalobj[symblname] = day_max_val
                tmpidobj[symblname] = max_bbandid[i2]
            target_idlist.append(tmpidobj)
            target_vallist.append(tmpvalobj)
        return target_idlist, target_vallist

    def simu_gains_close_history(self, policy_config, strategy_config, predict_bars, target_list, f_ratio_json,
                                 date_range):
        # 只用收盘价的操作策略
        all_holdings = []
        all_positions = []
        all_ratios = []
        datalenth = len(f_ratio_json[predict_bars.symbol_list[0]][0])
        barlenth = len(predict_bars.symbol_ori_data[predict_bars.symbol_list[0]]["close"])
        date_from = date_range[0] if date_range[0] >= 0 else barlenth + date_range[0]
        for i1 in range(0, datalenth):
            d = {}
            d['datetime'] = i1
            d['cash'] = policy_config["initial_capital"]
            d['commission'] = 0.0
            d['total'] = policy_config["initial_capital"]
            all_holdings.append(d)
            v = dict((k, v) for k, v in [(s, 0) for s in predict_bars.symbol_list])
            all_positions.append(v)
            u = dict((k, v) for k, v in [(s, 0.0) for s in predict_bars.symbol_list])
            all_ratios.append(u)
        for id1, i1 in enumerate(target_list):
            key_list = list(i1.keys())
            try:
                key_list.remove("没有")
            except Exception as e:
                pass
            key_lenth = len(key_list)
            print("id1:", id1, i1, key_list)
            if id1 == 0:
                continue
            # 1. 更新今日的资产 临时用cash存放
            price_cid = date_from + all_holdings[id1]["datetime"]
            all_holdings[id1]["cash"] = all_holdings[id1 - 1]["cash"]
            for i2 in predict_bars.symbol_list:
                price_c = predict_bars.symbol_ori_data[i2]["close"][price_cid]
                mount_pre = all_positions[id1 - 1][i2]
                # 清空折现
                if mount_pre > 0:
                    # 大于零时 平仓
                    all_holdings[id1]["cash"] += mount_pre * policy_config["hand_unit"] * price_c
                elif mount_pre < 0:
                    # 小于零时 平仓
                    all_holdings[id1]["cash"] += -mount_pre * policy_config["hand_unit"] * price_c
                    # 今日清空
                all_positions[id1][i2] = 0
            # 2. 重赋总资产
            all_holdings[id1]["total"] = all_holdings[id1]["cash"]
            # 根据比例，仓位计算
            for i2 in predict_bars.symbol_list:
                # price_c 为 nan 时不应该成为 标的
                price_c = predict_bars.symbol_ori_data[i2]["close"][price_cid]
                # 数据结构 已支持多标的综合操作
                for key1 in key_list:
                    if i2 == key1:
                        if key1 in predict_bars.symbol_list:
                            f_ratio_c = f_ratio_json[i2][i1[i2]][id1]
                        else:
                            f_ratio_c = 0
                        # 在列表中加仓
                        all_ratios[id1][i2] = float(f_ratio_c)
                        targ_captail = f_ratio_c * all_holdings[id1]["total"] / key_lenth
                        targ_mount = targ_captail / policy_config["hand_unit"] // price_c
                        all_positions[id1][i2] = targ_mount
            # 4. 加载 今日 头寸
            for i2 in predict_bars.symbol_list:
                price_c = predict_bars.symbol_ori_data[i2]["close"][price_cid]
                # print(i2, price_c)
                mount_pre = all_positions[id1 - 1][i2]
                mount_c = all_positions[id1][i2]
                if i2 not in key_list:
                    if mount_pre == 0:
                        # 不在目标列表 且 没有仓位继续循环
                        continue
                    else:
                        # 不在目标列表 且 有仓位 清空 放弃筹码
                        print("仓位 清空")
                        fees = self.calcu_fee(policy_config["commission"], policy_config["commission_rate"],
                                              policy_config["stamp_tax_in"], policy_config["stamp_tax_out"],
                                              0, mount_pre, price_c, policy_config["hand_unit"])
                        all_holdings[id1]["commission"] += fees
                        all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c + fees
                        all_holdings[id1]["total"] -= fees
                else:
                    # 在目标列表 增减到目标头寸
                    f_ratio_c = f_ratio_json[i2][i1[i2]][id1]
                    print("目标 为 {},且 mount_c !=0 : {}, price_now 为 {}, f_ratio才是 {}".format(i2, mount_c, price_c,
                                                                                            f_ratio_c))
                    if f_ratio_c > 0:
                        mount_add = mount_c - mount_pre
                        if mount_add > 0:
                            print("仓位 增加")
                            fees = self.calcu_fee(policy_config["commission"], policy_config["commission_rate"],
                                                  policy_config["stamp_tax_in"], policy_config["stamp_tax_out"],
                                                  mount_add, 0, price_c, policy_config["hand_unit"])
                            all_holdings[id1]["commission"] += fees
                            all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c + fees
                            all_holdings[id1]["total"] -= fees
                        elif mount_add < 0:
                            print("仓位 减仓")
                            fees = self.calcu_fee(policy_config["commission"], policy_config["commission_rate"],
                                                  policy_config["stamp_tax_in"], policy_config["stamp_tax_out"],
                                                  0, -mount_add, price_c, policy_config["hand_unit"])
                            all_holdings[id1]["commission"] += fees
                            all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c + fees
                            all_holdings[id1]["total"] -= fees
                        else:
                            print("仓位 不变")
                            all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c
            print(all_holdings[id1],
                  ["{}:{}".format(i2, all_positions[id1][i2]) for i2 in all_positions[id1] if
                   all_positions[id1][i2] != 0],
                  ["{}:{}".format(i2, all_ratios[id1][i2]) for i2 in all_ratios[id1] if all_positions[id1][i2] != 0])
        annual_ratio = math.pow(all_holdings[-1]["total"] / policy_config["initial_capital"], 252 / datalenth)
        print("annual_ratio: 1d {}, 1w {}, 1m {}, 1y {}, 2y {}, 4y {}, 8y {}, 10y {}"
              "".format(math.pow(annual_ratio, 1 / 252),
                        math.pow(annual_ratio, 5 / 252),
                        math.pow(annual_ratio, 22 / 252),
                        annual_ratio,
                        math.pow(annual_ratio, 2),
                        math.pow(annual_ratio, 4),
                        math.pow(annual_ratio, 8),
                        math.pow(annual_ratio, 10)))
        return all_holdings, all_positions, all_ratios

    def simu_gains_every_history(self, policy_config, strategy_config, predict_bars, pred_list_json, target_list,
                                 gain_json, f_ratio_json, date_range):
        all_holdings = []
        all_positions = []
        all_ratios = []
        all_oper_price = []
        # todo: 1. 为什么每次随机？ 2. 起始计数归一化。
        datalenth = len(f_ratio_json[predict_bars.symbol_list[0]][0])
        barlenth = len(predict_bars.symbol_ori_data[predict_bars.symbol_list[0]]["close"])
        date_from = date_range[0] if date_range[0] >= 0 else barlenth + date_range[0]
        # 1. 初始化循环每一天
        for i1 in range(0, datalenth):
            d = {}
            d['datetime'] = i1
            d['cash'] = policy_config["initial_capital"]
            d['commission'] = 0.0
            d['total'] = policy_config["initial_capital"]
            all_holdings.append(d)
            v = dict((k, v) for k, v in [(s, 0) for s in predict_bars.symbol_list])
            all_positions.append(v)
            u = dict((k, v) for k, v in [(s, 0.0) for s in predict_bars.symbol_list])
            all_ratios.append(u)
            all_oper_price.append({})
        # 2. 模拟循环每一天
        # pred_list: y_reta, y_reth, y_retl, y_stdup, y_stddw, y_drawup, y_drawdw
        move_in_percent = strategy_config["move_in_percent"]
        move_out_percent = strategy_config["move_out_percent"]
        for id1, key_bbandjson in enumerate(target_list):
            key_list = list(key_bbandjson.keys())
            try:
                key_list.remove("没有")
            except Exception as e:
                pass
            key_lenth = len(key_list)
            print("id1:", id1, key_bbandjson)
            if id1 == 0:
                continue
            # 2.1. 更新今日的资产 临时用cash存放，生成模拟的当天操作价位
            price_cid = date_from + all_holdings[id1]["datetime"]
            all_holdings[id1]["cash"] = all_holdings[id1 - 1]["cash"]
            price_c_in_json = {}
            price_c_out_json = {}
            pre_key_bbandjson = target_list[id1 - 1]
            pre_key_list = list(pre_key_bbandjson.keys())
            day2_bbandjson = copy.deepcopy(pre_key_bbandjson)
            day2_bbandjson.update(key_bbandjson)
            oper_symbol = set(day2_bbandjson.keys())
            try:
                oper_symbol.remove("没有")
            except Exception as e:
                pass
            # 只用昨天和今天的symbol
            for i2 in oper_symbol:
                # 无法保证时序，只能近似 生成操作价位
                # price_c 为 nan 时不应该成为 标的
                price_c = predict_bars.symbol_ori_data[i2]["close"][price_cid]
                price_c_h = predict_bars.symbol_ori_data[i2]["high"][price_cid]
                price_c_l = predict_bars.symbol_ori_data[i2]["low"][price_cid]
                price_pre = predict_bars.symbol_ori_data[i2]["close"][price_cid - 1]
                mount_pre = all_positions[id1 - 1][i2]
                # pred_list: y_reta, y_reth, y_retl, y_stdup, y_stddw, y_drawup, y_drawdw
                tmpmaxcol = np.argmax([i3[id1 - 1] for i3 in gain_json[i2]])
                pred_price_c = pred_list_json[i2][0][id1 - 1, tmpmaxcol]
                pred_price_h = pred_list_json[i2][1][id1 - 1, tmpmaxcol]
                pred_price_l = pred_list_json[i2][2][id1 - 1, tmpmaxcol]
                #
                ideal_outprice = price_pre * (pred_price_c + (pred_price_h - pred_price_c) * move_out_percent)
                move_out_price = price_c if ideal_outprice > price_c_h else ideal_outprice
                ideal_inprice = price_pre * (pred_price_l + (pred_price_c - pred_price_l) * move_in_percent)
                move_in_price = price_c if ideal_inprice < price_c_l else ideal_inprice
                price_c_out_json[i2] = move_out_price
                price_c_in_json[i2] = move_in_price
                # 清空折现
                if mount_pre > 0:
                    # 大于零时 平仓
                    all_holdings[id1]["cash"] += mount_pre * policy_config["hand_unit"] * price_c_out_json[i2]
                elif mount_pre < 0:
                    # 小于零时 平仓
                    all_holdings[id1]["cash"] += -mount_pre * policy_config["hand_unit"] * price_c_in_json[i2]
                    # 今日清空
                all_positions[id1][i2] = 0
            # 2.2 重赋总资产
            all_holdings[id1]["total"] = all_holdings[id1]["cash"]
            # 根据比例，仓位计算 all_positions all_ratios
            for i2 in oper_symbol:
                # 数据结构 已支持多标的综合操作
                for key1 in key_list:
                    if i2 == key1:
                        if key1 in predict_bars.symbol_list:
                            f_ratio_c = f_ratio_json[i2][day2_bbandjson[i2]][id1]
                        else:
                            f_ratio_c = 0
                        # 在列表中加仓
                        all_ratios[id1][i2] = float(f_ratio_c)
                        targ_captail = f_ratio_c * all_holdings[id1]["total"] / key_lenth
                        targ_mount = targ_captail / policy_config["hand_unit"] // price_c_in_json[i2]
                        all_positions[id1][i2] = targ_mount
            # 2.3. 更新核心 加载 今日 头寸
            for i2 in oper_symbol:
                price_pre = predict_bars.symbol_ori_data[i2]["close"][price_cid - 1]
                price_c = predict_bars.symbol_ori_data[i2]["close"][price_cid]
                price_h = predict_bars.symbol_ori_data[i2]["high"][price_cid]
                price_l = predict_bars.symbol_ori_data[i2]["low"][price_cid]
                # pred_list: y_reta, y_reth, y_retl, y_stdup, y_stddw, y_drawup, y_drawdw
                tmpmaxcol = np.argmax([i3[id1 - 1] for i3 in gain_json[i2]])
                pred_price_c = pred_list_json[i2][0][id1 - 1, tmpmaxcol]
                pred_price_h = pred_list_json[i2][1][id1 - 1, tmpmaxcol]
                pred_price_l = pred_list_json[i2][2][id1 - 1, tmpmaxcol]
                mount_pre = all_positions[id1 - 1][i2]
                mount_c = all_positions[id1][i2]
                ideal_outprice = price_pre * (pred_price_c + (pred_price_h - pred_price_c) * move_out_percent)
                ideal_inprice = price_pre * (pred_price_l + (pred_price_c - pred_price_l) * move_in_percent)
                all_oper_price[id1][i2] = {"pre_high": float(pred_price_h), "pre_low": float(pred_price_l),
                                           "pre_close": float(pred_price_c), "price_pre": price_pre,
                                           "pre_high_ins": float(pred_price_h) * price_pre,
                                           "pre_low_ins": float(pred_price_l) * price_pre,
                                           "pre_close_ins": float(pred_price_c) * price_pre,
                                           "in": float(ideal_inprice), "out": float(ideal_outprice)}
                print("目标：{}, mount_c: {}, mount_pre: {}, real_c {},real_h {},real_l {}, "
                      "pred_c {},pred_h {},pred_l {}".format(i2, mount_c, mount_pre, price_c, price_h, price_l,
                                                             pred_price_c * price_pre, pred_price_h * price_pre,
                                                             pred_price_l * price_pre))
                if i2 not in key_list:
                    if mount_pre == 0:
                        # 不在目标列表 且 没有仓位继续循环
                        continue
                    else:
                        # 不在目标列表 且 有仓位 清空 放弃筹码
                        print("仓位 清空 {}:{}".format(i2, price_c_out_json[i2]))
                        fees = self.calcu_fee(policy_config["commission"], policy_config["commission_rate"],
                                              policy_config["stamp_tax_in"], policy_config["stamp_tax_out"],
                                              0, mount_pre, price_c_out_json[i2], policy_config["hand_unit"])
                        all_holdings[id1]["commission"] += fees
                        all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c_out_json[i2] + fees
                        all_holdings[id1]["total"] -= fees
                else:
                    # 在目标列表 增减到目标头寸
                    f_ratio_c = f_ratio_json[i2][day2_bbandjson[i2]][id1]
                    print("f_ratio:{}".format(f_ratio_c))
                    if f_ratio_c > 0:
                        mount_add = mount_c - mount_pre
                        if mount_add > 0:
                            print("仓位 增加 {}:{}".format(i2, price_c_in_json[i2]))
                            fees = self.calcu_fee(policy_config["commission"], policy_config["commission_rate"],
                                                  policy_config["stamp_tax_in"], policy_config["stamp_tax_out"],
                                                  mount_add, 0, price_c_in_json[i2], policy_config["hand_unit"])
                            all_holdings[id1]["commission"] += fees
                            all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c_in_json[
                                i2] + fees
                            all_holdings[id1]["total"] -= fees
                        elif mount_add < 0:
                            print("仓位 减仓 {}:{}".format(i2, price_c_out_json[i2]))
                            fees = self.calcu_fee(policy_config["commission"], policy_config["commission_rate"],
                                                  policy_config["stamp_tax_in"], policy_config["stamp_tax_out"],
                                                  0, -mount_add, price_c_out_json[i2], policy_config["hand_unit"])
                            all_holdings[id1]["commission"] += fees
                            all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c_out_json[
                                i2] + fees
                            all_holdings[id1]["total"] -= fees
                        else:
                            print("仓位 不变 {}:{}".format(i2, price_c))
                            all_holdings[id1]["cash"] -= mount_c * policy_config["hand_unit"] * price_c
            print(all_holdings[id1],
                  ["{}:{}".format(i2, all_positions[id1][i2]) for i2 in all_positions[id1] if
                   all_positions[id1][i2] != 0],
                  ["{}:{}".format(i2, all_ratios[id1][i2]) for i2 in all_ratios[id1] if all_positions[id1][i2] != 0])
        annual_ratio = math.pow(all_holdings[-1]["total"] / policy_config["initial_capital"], 252 / datalenth)
        print("annual_ratio: 1d {}, 1w {}, 1m {}, 1y {}, 2y {}, 4y {}, 8y {}, 10y {}"
              "".format(math.pow(annual_ratio, 1 / 252),
                        math.pow(annual_ratio, 5 / 252),
                        math.pow(annual_ratio, 22 / 252),
                        annual_ratio,
                        math.pow(annual_ratio, 2),
                        math.pow(annual_ratio, 4),
                        math.pow(annual_ratio, 8),
                        math.pow(annual_ratio, 10)))
        return all_holdings, all_positions, all_ratios, all_oper_price

    # 基于预测结果 盈利测试
    def calculate_probability_every_signals(self, bband_list, pred_list):
        """kelly_formula"""
        system_risk = 0.0001  # 系统归零概率
        system_move = 1  # 上下概率平移系数
        system_ram_vari = 1  # 振幅系数

        upprb = []
        downprb = []
        f_ratio = []
        gain = []
        inst = TradeTool()
        llenth = len(bband_list)
        for id1, aven in enumerate(bband_list):
            fixratio = system_ram_vari / aven
            fw = pred_list[1][:, id1] - 1.0
            fw = fw * fixratio
            fl = 1.0 - pred_list[2][:, id1]
            fl = fl * fixratio
            upconst = np.exp(-(fw * system_move) ** 2 / pred_list[3][:, id1] ** 2)
            downconst = np.exp(-(fl / system_move) ** 2 / pred_list[4][:, id1] ** 2)
            # 加入系统风险后的极值一阶导数方程： y = a*f_ratio^2+b*f_ratio+c
            p = (1 - system_risk) * upconst / (upconst + downconst)
            q = (1 - system_risk) * downconst / (upconst + downconst)
            wm = inst.kari_fix_normal_w(p, q, fw, fl, system_risk)
            wm[:][wm[:] < 0] = 0
            wm[:][fw[:] < 0] = 0
            wm[:][fl[:] < 0] = 0
            upprb.append(p)
            downprb.append(q)
            f_ratio.append(wm)
            # print(p, q, fw, fl, system_risk, wm)
            gain.append(inst.kari_fix_normal_g(p, q, fw, fl, system_risk, wm))
            # print(list(zip(p, q, fw, fl, wm, gain[-1])))
        return upprb, downprb, f_ratio, gain
