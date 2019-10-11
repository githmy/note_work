from __future__ import print_function
import queue
import pprint
import time
import json
from modules.event import *
from utils.log_tool import *
from utils.mlp_tool import PlotTool
from utils.email_tool import email_info
from modules.datahandle import LoadCSVHandler


# import numpy as np


class LoadBacktest(object):
    def __init__(self, initial_capital, heartbeat, start_date,
                 csv_dir, symbol_list, ave_list, bband_list,
                 data_handler_cls, execution_handler_cls, portfolio_cls, strategy_cls,
                 split=0.8, newdata=0, date_range=[1, None], assistant=[]):
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.ave_list = ave_list
        self.bband_list = bband_list
        self.email_list = assistant
        self.data_handler_cls = data_handler_cls
        self.execution_handler_cls = execution_handler_cls
        self.portfolio_cls = portfolio_cls
        self.strategy_cls = strategy_cls

        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.newdata = newdata
        self.split = split
        self.date_range = date_range
        self._generate_trading_instances()
        print(self.symbol_list)
        print(len(self.symbol_list))
        print(self.ave_list)
        print(self.bband_list)

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

    def train(self):
        # self.symbol_pre_half_std_up
        # self.symbol_pre_half_std_down
        # self.symbol_pre_retp
        # self.symbol_pre_retm

        # self.symbol_aft_reta
        # self.symbol_aft_retp_high
        # self.symbol_aft_retp_low
        # self.symbol_aft_half_std_up
        # self.symbol_aft_half_std_down
        # self.symbol_aft_drawup
        # self.symbol_aft_drawdown
        # 1. 训练数据, 输入原始规范训练数据，待时间截断
        # todo: 1. 测试接口更新数据 2. 选出合理的bband 3. 用bband组合再次训练 预测
        train_bars = LoadCSVHandler(queue.Queue(), data_path, self.symbol_list, self.ave_list, self.bband_list)
        # date_range = [1, None]
        # split = 0.8  # 先截range 再split
        # 3. 训练
        self._strategy.train_probability_everysignals(train_bars, self.ave_list, self.bband_list, self.date_range,
                                                      newdata=self.newdata, split=self.split, args=None)

    # 回测，根据不同事件执行不同的方法
    def _run_backtest(self, policy_config, strategy_config, startdate="2000-01-01 00:00:00"):
        print("in backtest from")
        predict_bars_json = {}
        pred_list_json = {}
        # date_range = [550, None]
        # date_range = [0, None]
        predict_bars = LoadCSVHandler(queue.Queue(), data_path, self.symbol_list, self.ave_list, self.bband_list)
        print(startdate)
        if startdate is not None:
            predict_bars.get_some_net_csv2files(startdate=startdate)
            predict_bars.get_some_current_net_csv2files()
        predict_bars.generate_b_derivative()
        # 2. 预测投资比例
        print("data full lenth: {}".format(len(predict_bars.symbol_ori_data[self.symbol_list[0]]["close"].index)))
        pred_list_json = self._strategy.predict_probability_signals(predict_bars, self.ave_list, self.bband_list,
                                                                    self.date_range, args=None)
        print("data used lenth: {}".format(len(pred_list_json[self.symbol_list[0]][0])))
        # print(pred_list_json)
        # 3. 投资回测结果 此处不需要指定date range 因为预测数据已经在此范围内
        all_holdings, all_positions, all_ratios = self._portfolio.components_res_every_predict(predict_bars,
                                                                                               pred_list_json,
                                                                                               policy_config,
                                                                                               strategy_config,
                                                                                               self.date_range)
        # 4. 绘制收益过程
        show_list = []
        show_x = [i1["datetime"] for i1 in all_holdings]
        show_y = [i1["total"] for i1 in all_holdings]
        show_list.append([show_x, show_y])
        titie_str = "gain curve"
        titie_list = [titie_str, *predict_bars.symbol_list]
        for symbol in predict_bars.symbol_list:
            tmp_ori = predict_bars.symbol_ori_data[symbol]["close"].values[self.date_range[0]:self.date_range[1]]
            tmp_y = tmp_ori * (policy_config["initial_capital"] / tmp_ori[0])
            show_list.append([show_x, tmp_y])
        insplt = PlotTool()
        insplt.plot_line(show_list, titie_list)
        # 5. 返回邮件数据
        return all_holdings[-1], all_positions[-1], all_ratios[-1]

    # 从回测中得到策略的表现
    def _output_performance(self):
        """输出 回测的 性能结果"""
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
    def simulate_trading(self, policy_config, strategy_config, startdate="2000-01-01 00:00:00"):
        """回测 输出组合的 性能"""
        contents = self._run_backtest(policy_config, strategy_config, startdate=startdate)
        # self._output_performance()
        return None
        strs_list = []
        for i1 in contents:
            tmpjson = {}
            for i2 in i1:
                if not (i1[i2] == 0 or i1[i2] == 0.0):
                    tmpjson[i2] = i1[i2]
            strs_list.append(json.dumps(tmpjson, indent=4, ensure_ascii=False))
        # 发送邮件
        headstr = "策略信息"
        email_info(headstr, strs_list, addresses=self.email_list)

    # 模拟回测最后一天的不同情况
    def simulate_lastday(self, policy_config, showconfig, startdate="2000-01-01 00:00:00"):
        """回测 最后一天的不同情况"""
        pred_list_json = {}
        # 1. 预测概率
        predict_bars = LoadCSVHandler(queue.Queue(), data_path, self.symbol_list, self.ave_list, self.bband_list)
        if startdate is not None:
            predict_bars.get_some_net_csv2files(startdate=startdate)
        predict_bars.generate_b_derivative()
        # 2. 预测投资比例
        pred_list_json, fake_ori = self._strategy.predict_fake_proba_signals(predict_bars, self.ave_list,
                                                                             self.bband_list, showconfig, args=None)
        # 3. 虚拟价格的操作空间
        fake_gain, fake_f_ratio, fake_mount = self._portfolio.components_res_fake_predict(predict_bars, pred_list_json,
                                                                                          fake_ori, policy_config)
        print("fake_gain-", fake_gain)
        print("fake_f_ratio-", fake_f_ratio)
        print("fake_mount-", fake_mount)
        # 4. 绘制收益过程
        insplt = PlotTool()
        for symbol in self.symbol_list:
            avestr = ",".join([str(i2) for i2 in self.bband_list])
            titie_str = "gain {} ave {}".format(symbol, avestr)
            insplt.plot_dim3(fake_gain[symbol], titie_str, **showconfig)
            titie_str = "f_ratio {} ave {}".format(symbol, avestr)
            insplt.plot_dim3(fake_f_ratio[symbol], titie_str, **showconfig)
            titie_str = "keep_mount {} ave {}".format(symbol, avestr)
            insplt.plot_dim3([fake_mount[symbol]], titie_str, **showconfig)


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
        # self._output_performance()
