import sys
import time
import queue
import datetime
import os, os.path
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
from modules.portfolio import Portfolio
from pyalgotrade import strategy
from utils.log_tool import *
import numpy as np

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
    def __init__(self, events, csv_dir, symbol_list):
        self.b_continue_backtest = True
        self.events = events
        # symbol_list:传入要处理的symbol列表集合，list类型
        self.symbol_list = symbol_list
        self.symbol_list_with_benchmark = copy.deepcopy(self.symbol_list)
        # self.symbol_list_with_benchmark.append('000300')
        self.csv_dir = csv_dir
        self.symbol_data = {}  # symbol_data，{symbol:DataFrame}
        self.latest_symbol_data = {}  # 最新的bar:{symbol:[bar1,bar2,barNew]}

        self.b_continue_backtest = True
        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        comb_index = None
        for s in self.symbol_list_with_benchmark:
            # 加载csv文件,date,OHLC,Volume
            self.symbol_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s), header=0, index_col=0, parse_dates=False,
                names=['date', 'open', 'high', 'low', 'close', 'volume']).sort_index()

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                # 这里要赋值，否则comb_index还是原来的index
                comb_index = comb_index.union(self.symbol_data[s].index)
        # 设置latest symbol_data 为 None
        self.latest_symbol_data[s] = []
        # Reindex the dataframes
        for s in self.symbol_list_with_benchmark:
            # 这是一个发生器iterrows[index,series],用next(self.symbol_data[s])
            # pad方式，就是用前一天的数据再填充这一天的丢失，对于资本市场这是合理的，比如这段时间停牌。那就是按停牌前一天的价格数据来计算。
            self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        row = (index,series),row[0]=index,row[1]=[OHLCV]
        """
        row = next(self.symbol_data[symbol])
        # return tuple(symbol,row[0],row[1][0],row[1][1],row[1][2],row[1][3],row[1][4]])
        row_dict = {'symbol': symbol, 'date': row[0], 'open': row[1][0], 'high': row[1][1], 'low': row[1][2],
                    'close': row[1][3]}
        return row_dict

    def update_bars(self):
        """
        循环每只股票的下一个值
        """
        for s in self.symbol_list_with_benchmark:
            try:
                bar = self._get_new_bar(s)
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
        else:
            return bars_list[-N:]

    def get_latest_bars(self, symbol, N=1):
        """
        返回最近的 N bars 或 不够时 N-k 个.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        print("not implyment")


# order与fill之间的交互基类，可用于实际或模拟成交。
class ExecutionHandler(object):
    # 获取Order event并执行，产生Fill Event并放到队列
    def execute_order(self, event):
        pass


# 模拟执行所有order object转为成交对象，不考虑延时，滑价和成交比率影响
class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, events):
        self.events = events

    # 和该类描述一致
    def execute_order(self, event):
        if event.type == 'ORDER':
            fill_event = FillEvent(datetime.datetime.utcnow(), event.symbol, 'ARCA', event.quantity, event.direction,
                                   None)
            self.events.put(fill_event)


# 封装对数据的计算，并且生成相应的信号:  策略处理基类，可用于处理历史和实际交易数据，只需把数据存到队列中。
# 移动平均跨越策略。用短期/长期移动平均值进行基本的移动平均跨越的实现。
class MovingAverageCrossStrategy(strategy.BacktestingStrategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 100/400 periods respectively.
    """

    def __init__(self, bars, events, short_window=100, long_window=400):
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
        Generates a new set of signals based on the MAC
        SMA with the short window crossing the long window
        meaning a long entry and vice versa for a short entry.    

        Parameters
        event - A MarketEvent object. 
        """
        if event.type == 'MARKET':
            for symbol in self.symbol_list:
                bars = self.bars.get_latest_bars_values(symbol, "close", N=self.long_window)

                if bars is not None and bars != []:
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])

                    dt = self.bars.get_latest_bar_datetime(symbol)
                    sig_dir = ""
                    strength = 1.0
                    strategy_id = 1

                    if short_sma > long_sma and self.bought[symbol] == "OUT":
                        sig_dir = 'LONG'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'

                    elif short_sma < long_sma and self.bought[symbol] == "LONG":
                        sig_dir = 'EXIT'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'OUT'


"""
BarEvent - 新的tick，也就是新的Bar（可以认为就是一个K线，这里简单起见就用日线，换成5分钟线，小时线类似）到达。
   当DataHandler.update_ars更新时触发，
   用于Strategy计算交易信号，P
   ortfolio更新仓位信息。
   BarEvent只需要一个type，没有其他的成员变量。
SignalEvent -信号事件。
   Strategy在处理每天的Bar时，如果按模型计算，需要产生信号时，触发Signal事件，包含对特定symbol做多，做空或平仓。
   SignalEvent被Porfolio对象用于计算如何交易。
OrderEvent - 订单事件。
   当Porfolio对象接受一件SignalEvent事件时，会根据当前的风险和仓位，
   发现OrderEvent给ExecutionHandler执行器。
FillEvent - 交易订单。
   当执行器ExecutionHandler接收到OrderEvent后就会执行交易订单，
   订单交易完成时会产生FillEvent，
   给Porfolio对象去更新成本，仓位情况等操作。
"""


# 事件基类，提供子类的一个接口
class Event(object):
    pass


class BarEvent(Event):
    def __init__(self):
        self.type = 'BAR'


# 处理市场数据更新，触发Strategy生成交易信号。
class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with 
    corresponding bars.
    """

    def __init__(self):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'


# 处理Strategy发来的信号，信号会被Portfolilo 接收和执行
class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """

    def __init__(self, strategy_id, symbol, datetime, signal_type, strength):
        """
        Initialises the SignalEvent.

        Parameters:
        strategy_id - The unique ID of the strategy sending the signal.
        symbol - The ticker symbol, e.g. 'GOOG'.
        datetime - The timestamp at which the signal was generated.
        signal_type - 'LONG' or 'SHORT'.
        strength - An adjustment factor "suggestion" used to scale 
            quantity at the portfolio level. Useful for pairs strategies.
        """
        self.strategy_id = strategy_id
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength


# 处理向执行系统提交的订单信息
class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self, symbol, order_type, quantity, direction):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral) and its direction ('BUY' or
        'SELL').

        TODO: Must handle error checking here to obtain 
        rational orders (i.e. no negative quantities etc).

        Parameters:
        symbol - The instrument to trade.
        order_type - 'MKT' or 'LMT' for Market or Limit.
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL' for long or short.
        """
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        print("Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s" % (
            self.symbol, self.order_type, self.quantity, self.direction))


# 封装订单执行。存储交易数量、价格、佣金和手续费。
class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.

    TODO: Currently does not support filling positions at
    different prices. This will be simulated by averaging
    the cost.
    """

    def __init__(self, timeindex, symbol, exchange, quantity,
                 direction, fill_cost, commission=None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional 
        commission.

        If commission is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.

        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL')
        fill_cost - The holdings value in dollars.
        commission - An optional commission sent from IB.
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost

        # Calculate commission
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    # 计算Interactive Brokers 交易费用
    def calculate_ib_commission(self):
        """
        Calculates the fees of trading based on an Interactive
        Brokers fee structure for API, in USD.

        This does not include exchange or ECN fees.

        Based on "US API Directed Orders":
        https://www.interactivebrokers.com/en/index.php?f=commission&p=stocks2
        """
        full_cost = 1.3
        if self.quantity <= 500:
            full_cost = max(1.3, 0.013 * self.quantity)
        else:  # Greater than 500
            full_cost = max(1.3, 0.008 * self.quantity)
        return full_cost


# 这个类进行驱动回测设置与组成
class Backtest(object):
    def __init__(self, csv_dir, csv_list, initial_capital, heartbeat, start_date,
                 data_handler_cls, execution_handler_cls, portfolio_cls, strategy_cls):
        self.csv_dir = csv_dir
        self.csv_list = csv_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.data_handler_cls = data_handler_cls
        self.execution_handler_cls = execution_handler_cls
        self.portfolio_cls = portfolio_cls
        self.strategy_cls = strategy_cls

        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self._generate_trading_instances()

    # 从不同类型产生交易实例
    def _generate_trading_instances(self):
        """
        按策略执行后的结果
        Generates the trading instance objects from their class types.
        """
        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        self._data_handler = self.data_handler_cls(self.events, self.csv_dir, self.csv_list)
        self._portfolio = self.portfolio_cls(self._data_handler, self.events, self.start_date, self.initial_capital)
        self._execution = self.execution_handler_cls(self.events)
        self._strategy = self.strategy_cls(self._data_handler, self.events)
        self._init_event_handlers()

    # 挂载事件，每个事件类型对应一个处理函数
    def _init_event_handlers(self):
        self._event_handler = {}
        self._event_handler['BAR'] = self._handle_event_bar
        self._event_handler['SIGNAL'] = self._handle_event_signal
        self._event_handler['ORDER'] = self._handle_event_order
        self._event_handler['FILL'] = self._handle_event_fill

    # 回测，根据不同事件执行不同的方法
    def _run_backtest(self):
        i = 0
        while True:
            i += 1
            print(i)
            # 每个时间段 loop，都会让data_handler触发新的bar事件，即有一个新bar到达
            # b_continue_backtest是数据管理器的标志，置False则回测停止
            if self._data_handler.b_continue_backtest:
                self._data_handler.update_bars()
            else:
                break  # 可能有多事件，要处理完，如果队列暂时是空的，不处理
            while True:
                # 队列为空则消息循环结束
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    self._handle_event(event)
            time.sleep(self.heartbeat)

    # 从回测中得到策略的表现
    def _output_performance(self):
        """
        输出 回测的 性能结果
        """
        self.portfolio.create_equity_curve_dataframe()

        print("输出摘要统计信息")
        stats = self.portfolio.output_summary_stats()

        print("输出股本曲线")
        print(self.portfolio.equity_curve.tail(10))
        pprint.pprint(stats)

        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)

    # 模拟回测并输出投资组合表现
    def simulate_trading(self):
        """
        执行输出 回测和组合的 性能
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        self._output_performance()

    def show_results(self):
        self._portfolio.output_summary_stats()

    def _handle_event(self, event):
        handler = self._event_handler.get(event.type, None)
        if handler is None:
            print('type:%s,handler is None' % event.type)
        else:
            handler(event)

    # 更新 bar
    def _handle_event_bar(self, event):
        print('OnBar Event', event.type)
        self._strategy.onBars()
        self._portfolio.update_timeindex()

    # 更新 mark
    def _handle_event_mark(self, event):
        print('OnMark Event', event.type)
        pass

    # 处理策略产生的交易信号
    def _handle_event_signal(self, event):
        print('OnSignal Event', event.type)
        self._portfolio.on_signal(event)

    # 处理ORDER
    def _handle_event_order(self, event):
        print('OnOrder Event', event.type)
        self._execution.execute_order(event)

    # 处理FILL
    def _handle_event_fill(self, event):
        print('OnFill Event', event.type)
        self._portfolio.on_fill(event)


def main(paralist):
    logger.info(paralist)
    start_date = datetime.datetime(1990, 1, 1, 0, 0, 0)
    heartbeat = 0.0

    # 1. 起止 学习 回测 的三个时间
    csv_dir = data_path
    # csv_list = ["ChinaBank", "DalianRP", "SAPower"]
    csv_list = ["SAPower"]
    initial_capital = 10000.0
    backtest = Backtest(csv_dir, csv_list, initial_capital, heartbeat, start_date,
                        CSVDataHandler, SimulatedExecutionHandler, Portfolio, MovingAverageCrossStrategy)
    backtest.simulate_trading()

    # 2. 起止 学习 预测 的三个时间


if __name__ == "__main__":
    # csv_dir = REPLACE_WITH_YOUR_CSV_DIR_HERE

    logger.info("".center(100, "*"))
    logger.info("welcome to surfing".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    main(sys.argv[1:])
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
