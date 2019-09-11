from __future__ import print_function
import queue
import pprint
import time
from modules.event import *
from utils.log_tool import *
from utils.mlp_tool import PlotTool
from modules.datahandle import LoadCSVHandler


class LoadBacktest(object):
    def __init__(self, initial_capital, heartbeat, start_date,
                 csv_dir, symbol_list, ave_list, bband_list,
                 data_handler_cls, execution_handler_cls, portfolio_cls, strategy_cls):
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.ave_list = ave_list
        self.bband_list = bband_list

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
        self._data_handler = self.data_handler_cls(self.events, self.csv_dir, self.symbol_list, self.ave_list,
                                                   self.bband_list)
        self._execution = self.execution_handler_cls(self.events)
        self._strategy = self.strategy_cls(self._data_handler, self.events, self.ave_list, self.bband_list)
        self._portfolio = self.portfolio_cls(self._data_handler, self.events, self.start_date,
                                             self.ave_list, self.bband_list, self.initial_capital)

    # 回测，根据不同事件执行不同的方法
    def _run_backtest(self):
        print(self.symbol_list)
        print(self.ave_list)
        print(self.bband_list)
        ave_list, bband_list = self.ave_list, self.bband_list
        # 1. 加载衍生前值
        self._data_handler.generate_b_derivative()
        # 加载衍生后值
        self._data_handler.generate_a_derivative()
        # 统计倾向概率
        self._strategy.calculate_probability_signals("event")
        # # 2. 训练数据, 输入原始规范训练数据，待时间截断
        # # todo: 1. 改用 tushare 数据，1.1 测试接口更新数据 2. 分类板块，每个模型，时间段回测
        # train_bars = LoadCSVHandler(queue.Queue(), data_path, ["DalianRP", "ChinaBank"], ave_list, bband_list)
        # train_bars.generate_b_derivative()
        # train_bars.generate_a_derivative()
        # date_range = [1, 200]
        # split = 0.8  # 先截range 再split
        # self._strategy.train_probability_signals(train_bars, ave_list, bband_list, date_range, split=split, args=None)
        # exit(0)
        # 3. 预测概率
        predict_bars = LoadCSVHandler(queue.Queue(), data_path, ["SAPower"], ave_list, bband_list)
        predict_bars.generate_b_derivative()
        date_range = [1, 260]
        pred_list = self._strategy.predict_probability_signals(predict_bars, ave_list, bband_list, date_range,
                                                               args=None)
        # print(pred_list)
        # print(pred_list[0].shape, pred_list[1].shape)
        # 4. 投资比例
        # self._portfolio.components_res_base_aft()
        # symbol_aft_retp * bband 均线 * bband 涨幅

        # symbol_aft_reta   bband 涨幅
        # symbol_aft_half_std_up
        # symbol_aft_half_std_down

        # symbol_aft_drawup
        # symbol_aft_drawdown
        # symbol_aft_retp_high
        # symbol_aft_retp_low
        all_holdings = self._portfolio.components_res_base_predict(predict_bars, pred_list)
        show_x, show_y = [i1["datetime"] for i1 in all_holdings], [i1["total"] for i1 in all_holdings]
        insplt = PlotTool()
        insplt.plot_line([[show_x, show_y]])
        # print(self._portfolio.all_holdings)
        # print(self._portfolio.all_positions)
        # 按规则投资的变化结果

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
        # self._output_performance()


# 这个类进行驱动回测设置与组成
class Backtest(object):
    def __init__(self, initial_capital, heartbeat, start_date,
                 csv_dir, symbol_list, ave_list, mount_list, ret_list,
                 data_handler_cls, execution_handler_cls, portfolio_cls, strategy_cls):
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.ave_list = ave_list
        self.mount_list = mount_list
        self.ret_list = ret_list

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
        # self._data_handler = self.data_handler_cls(self.events, self.csv_dir, self.symbol_list, self.ave_list,
        #                                            self.bband_list,self.ret_list)
        self._data_handler = self.data_handler_cls(self.events, self.csv_dir, self.symbol_list, self.ave_list)
        self._portfolio = self.portfolio_cls(self._data_handler, self.events, self.start_date, self.initial_capital)
        self._execution = self.execution_handler_cls(self.events)
        self._strategy = self.strategy_cls(self._data_handler, self.events, self.ave_list)
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
            print('type: %s, handler is None' % event.type)
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
