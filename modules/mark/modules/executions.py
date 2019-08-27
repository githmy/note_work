from __future__ import print_function
import datetime
from modules.event import *


# order与fill之间的交互基类，可用于实际或模拟成交。
class ExecutionHandler(object):
    # 获取Order event并执行，产生Fill Event并放到队列
    def execute_order(self, event):
        pass


# 模拟执行所有order object转为成交对象，不考虑延时，滑价和成交比率影响
class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, events):
        self.events = events

    # 订单事件 -> 填单事件
    def execute_order(self, event):
        if event.type == 'ORDER':
            # 交易所名称 'ARCA'
            fill_event = FillEvent(datetime.datetime.utcnow(), event.symbol, 'ARCA', event.quantity, event.direction,
                                   None)
            self.events.put(fill_event)
