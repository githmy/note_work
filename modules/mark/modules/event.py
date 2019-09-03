from __future__ import print_function

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


class Event(object):
    """
    Event is base class providing an interface for all subsequent 
    (inherited) events, that will trigger further events in the 
    trading infrastructure.   
    """
    pass


# 处理市场数据更新，触发Strategy生成交易信号。
class BarEvent(Event):
    def __init__(self):
        self.type = 'BAR'


# 处理Strategy发来的信号，信号会被Portfolilo 接收和执行
class SignalEvent(Event):
    """
    处理Strategy发来的信号，信号会被Portfolilo 接收和执行
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


# 处理Strategy发来的信号，信号会被Portfolilo 接收和执行
class SignalTrainEvent(Event):
    """
    处理Strategy发来的信号，信号会被Portfolilo 接收和执行
    """

    def __init__(self, strategy_id, symbol, datetime, signal_type, strength):
        """
        Parameters:
        strategy_id - The unique ID of the strategy sending the signal.
        symbol - The ticker symbol, e.g. 'GOOG'.
        datetime - The timestamp at which the signal was generated.
        signal_type - [3,5,1]
        strength - An adjustment factor "suggestion" used to scale 
            quantity at the portfolio level. Useful for pairs strategies.
        """
        self.strategy_id = strategy_id
        self.type = 'SignalTrain'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength


# 处理向执行系统提交的订单信息
class OrderEvent(Event):
    """
    处理向执行系统提交的订单信息
    """

    def __init__(self, symbol, order_type, quantity, direction):
        """
        Parameters:
        symbol - The instrument to trade.
        order_type - 'MKT' or 'LMT' for Market or Limit. 市价单，限价单。
        quantity - Non-negative integer for quantity. 非负整数
        direction - 'BUY' or 'SELL' for long or short.
        """
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        """
        打印订单信息
        """
        print("Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s" % (
            self.symbol, self.order_type, self.quantity, self.direction))


# 封装订单执行。存储交易数量、价格、佣金和手续费。
class FillEvent(Event):
    """
    封装订单执行。存储交易数量、价格、佣金和手续费。
    TODO: Currently does not support filling positions at
    different prices. This will be simulated by averaging
    the cost.
    """

    def __init__(self, timeindex, symbol, exchange, quantity,
                 direction, fill_cost, commission=None):
        """
        If commission is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.

        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled. 交易所名称
        quantity - The filled quantity.  成交数量
        direction - The direction of fill ('BUY' or 'SELL')
        fill_cost - The holdings value in dollars. 
        commission - An optional commission sent from IB. 佣金
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
