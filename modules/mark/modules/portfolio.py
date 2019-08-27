from __future__ import print_function

import os
import pandas as pd
from utils.log_tool import *
from modules.event import FillEvent, OrderEvent
from modules.performance import create_sharpe_ratio, create_drawdowns


class Portfolio(object):
    """
    资产组合类，处理持仓和市场价值。postion DataFrame存储用时间做索引的持仓数量；
    holdings DataFrame存储特定时间索引对应代码的现金和总市场持仓价值，以及资产组合总量的百分比变化
    """

    def __init__(self, bars, events, start_date, initial_capital=100000.0):
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
