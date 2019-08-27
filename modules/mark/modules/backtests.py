from __future__ import print_function
import queue
import pprint
import time
from modules.event import *
from utils.log_tool import *


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

    # 各种实例生成
    def _generate_trading_instances(self):
        """
        各种实例生成，事件处理的字典。
        """
        logger.info("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        self._data_handler = self.data_handler_cls(self.events, self.csv_dir, self.csv_list)
        self._portfolio = self.portfolio_cls(self._data_handler, self.events, self.start_date, self.initial_capital)
        self._execution = self.execution_handler_cls(self.events)
        self._strategy = self.strategy_cls(self._data_handler, self.events)
        # 事件处理的字典
        self._init_event_handlers()

    # 挂载事件，每个事件类型对应一个处理函数
    def _init_event_handlers(self):
        self._event_handler = {}
        self._event_handler['BAR'] = self._handle_event_bar
        self._event_handler['SIGNAL'] = self._handle_event_signal
        self._event_handler['ORDER'] = self._handle_event_order
        self._event_handler['FILL'] = self._handle_event_fill

    def _handle_event(self, event):
        handler = self._event_handler.get(event.type, None)
        if handler is None:
            print('type:%s,handler is None' % event.type)
        else:
            handler(event)

    # 更新 bar
    def _handle_event_bar(self, event):
        print('handle Event', event.type)
        # 生成操作指标信号
        self._strategy.calculate_signals(event)
        # 更新 持有的 总资产
        self._portfolio.update_timeindex(event)

    # 处理策略产生的交易信号
    def _handle_event_signal(self, event):
        print('handle Event', event.type)
        self.signals += 1
        self._portfolio.update_signals(event)

    # 处理ORDER
    def _handle_event_order(self, event):
        print('handle Event', event.type)
        self.orders += 1
        self._execution.execute_order(event)

    # 处理FILL
    def _handle_event_fill(self, event):
        print('handle Event', event.type)
        self.fills += 1
        self._portfolio.update_fill(event)

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
        self._portfolio.create_equity_curve_dataframe()

        print("输出股本曲线")
        print(self._portfolio.equity_curve.tail(10))

        print("输出摘要统计信息")
        stats = self._portfolio.output_summary_stats()
        pprint.pprint(stats)

        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)

    # 模拟回测并输出投资组合表现
    def simulate_trading(self):
        """
        回测 输出组合的 性能
        """
        self._run_backtest()
        self._output_performance()
