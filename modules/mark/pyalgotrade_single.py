# -*- coding: utf-8 -*-
from pyalgotrade import strategy
from pyalgotrade import technical
from pyalgotrade import plotter
from pyalgotrade.stratanalyzer import returns, sharpe, drawdown, trades
from pyalgotrade import bar
from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.barfeed import membf
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade import broker
import tushare as ts
import datetime
import itertools


def parse_date(date):
    # This custom parsing works faster than:
    # datetime.datetime.strptime(date, "%Y-%m-%d")
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    d = datetime.datetime(year, month, day)
    if len(date) > 10:
        h = int(date[11:13])
        m = int(date[14:16])
        t = datetime.time(h, m)
        ret = datetime.combine(d, t)
    else:
        ret = d
    return ret


class Feed(membf.BarFeed):
    def __init__(self, frequency=bar.Frequency.DAY, maxLen=None):
        super(Feed, self).__init__(frequency, maxLen)

    def rowParser(self, ds, frequency=bar.Frequency.DAY):
        dt = parse_date(ds["date"])
        open = float(ds["open"])
        close = float(ds["close"])
        high = float(ds["high"])
        low = float(ds["low"])
        volume = float(ds["volume"])
        return bar.BasicBar(dt, open, high, low, close, volume, None, frequency)

    def barsHaveAdjClose(self):
        return False

    def addBarsFromCode(self, code, start, end, ktype="D", index=False):
        frequency = bar.Frequency.DAY
        if ktype == "D":
            frequency = bar.Frequency.DAY
        elif ktype == "W":
            frequency = bar.Frequency.WEEK
        elif ktype == "M":
            frequency = bar.Frquency.MONTH
        elif ktype == "5" or ktype == "15" or ktype == "30" or ktype == "60":
            frequency = bar.Frequency.MINUTE
        ds = ts.get_k_data(code=code, start=start, end=end, ktype=ktype, index=index)
        bars_ = []
        for i in ds.index:
            bar_ = self.rowParser(ds.loc[i], frequency)
            bars_.append(bar_)

        self.addBarsFromSequence(code, bars_)


class DiffEventWindow(technical.EventWindow):
    def __init__(self, period):
        assert (period > 0)
        super(DiffEventWindow, self).__init__(period)
        self.__value = None

    def onNewValue(self, dateTime, value):
        super(DiffEventWindow, self).onNewValue(dateTime, value)
        if self.windowFull():
            lastValue = self.getValues()[0]
            nowValue = self.getValues()[1]
            self.__value = (nowValue - lastValue) / lastValue

    def getValue(self):
        return self.__value


class Diff(technical.EventBasedFilter):
    def __init__(self, dataSeries, period, maxLen=None):
        super(Diff, self).__init__(dataSeries, DiffEventWindow(period), maxLen)


class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, brk, diffPeriod=2):
        # super(MyStrategy, self).__init__(feed, 10000)
        super(MyStrategy, self).__init__(feed, brk)
        self.__instrument = instrument
        self.__position = None
        self.__rsi = rsi.RSI(feed[instrument].getCloseDataSeries(), 14)
        # technicals可以进行组合 如下：
        # self.__sma = ma.SMA(self.__rsi, 15)
        # 15 天均线
        self.__sma = ma.SMA(feed[instrument].getCloseDataSeries(), 15)
        # 用调整 收价 代替 普通收价
        if feed.barsHaveAdjClose():
            self.setUseAdjustedValues(True)
        self.__prices = feed[instrument].getPriceDataSeries()
        self.__diff = Diff(self.__prices, diffPeriod)
        self.__break = 0.03
        self.__withdown = -0.03
        self.getBroker()

    def getDiff(self):
        return self.__diff

    def getSMA(self):
        return self.__sma

    def onEnterCanceled(self, position):
        self.__position = None

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info("BUY at $%.2f" % (execInfo.getPrice()))

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info("SELL at $%.2f" % (execInfo.getPrice()))
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()

    def onBars(self, bars):
        # 交易规则
        # 1. 没有指标可用返回
        if self.__sma[-1] is None:
            return
        account = self.getBroker().getCash()
        bar = bars[self.__instrument]
        # 策略过程 打印盘价
        self.info("%s %s %s" % (bar.getClose(), self.__rsi[-1], self.__sma[-1]))
        # 2. 该位置是否可以交易
        if self.__position is None:
            if bar.getPrice() > self.__sma[-1]:
                # Enter a buy market order for 10 shares. The order is good till canceled.
                self.__position = self.enterLong(self.__instrument, 10, True)
        # Check if we have to exit the position.
        elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():
                self.__position.exitMarket()
        #     one = bar.getPrice() * 100
        #     oneUnit = account // one
        #     if oneUnit > 0 and self.__diff[-1] > self.__break:
        #         self.__position = self.enterLong(self.__instrument, oneUnit * 100, True)
        # elif self.__diff[-1] < self.__withdown and not self.__position.exitActive():
        #     self.__position.exitMarket()
        #     # SMA的计算存在窗口，所以前面的几个bar下是没有SMA的数据的.
        #     if self.__sma[-1] is None:
        #         return
        #         # bar.getTyoicalPrice = (bar.getHigh() + bar.getLow() + bar.getClose())/ 3.0
        #
        #     bar = bars[self.__instrument]
        #     # If a position was not opened, check if we should enter a long position.
        #     if self.__position is None:  # 如果手上没有头寸，那么
        #         if bar.getPrice() > self.__sma[-1]:
        #             # 开多，如果现价大于移动均线，且当前没有头寸.
        #             self.__position = self.enterLong(self.__instrument, 100, True)
        #             # 当前有多头头寸，平掉多头头寸.
        #     elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():
        #         self.__position.exitMarket()


def runStrategy():
    # # 1.下载数据
    # jdf = ts.get_k_data("000725")
    #
    # # 将tushare下的数据的字段保存为pyalgotrade所要求的数据格式
    # jdf.columns = ["Date", "Open", "Close", "High", "Low", "Volume", "code", "Adj Close"]
    #
    # # 将数据保存成本地csv文件
    # jdf.to_csv("jdf.csv", index=False)

    # 2.获得回测数据，
    code = "600001"
    feed = Feed()
    # feed = yahoofeed.Feed()
    # feed = GenericBarFeed(Frequency.DAY, None, None)
    # feed.addBarsFromCSV(code, code + ".csv")
    feed.addBarsFromCode(code, start='2008-01-29', end='2019-01-03')
    # for dateTime, value in feed:
    #     print(dateTime, value)
    # 3.broker setting
    # 3.1 commission类设置
    # a.没有手续费
    # broker_commission = pyalgotrade.broker.backtesting.NoCommission()
    # b.amount：每笔交易的手续费
    # broker_commission = pyalgotrade.broker.backtesting.FixedPerTrade(amount)
    # c.百分比手续费
    broker_commission = broker.backtesting.TradePercentage(0.0001)
    # 3.2 fill strategy设置
    fill_stra = broker.fillstrategy.DefaultStrategy(volumeLimit=0.1)
    sli_stra = broker.slippage.NoSlippage()
    fill_stra.setSlippageModel(sli_stra)
    # 3.3完善broker类
    brk = broker.backtesting.Broker(1000000, feed, broker_commission)
    brk.setFillStrategy(fill_stra)

    # 4.把策略跑起来
    myStrategy = MyStrategy(feed, code, brk)
    # Attach a returns analyzers to the strategy.
    trade_situation = trades.Trades()
    myStrategy.attachAnalyzer(trade_situation)

    returnsAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(returnsAnalyzer)

    # 5. Attach the plotter to the strategy.
    plt = plotter.StrategyPlotter(myStrategy)
    plt.getInstrumentSubplot(code)
    plt.getInstrumentSubplot(code).addDataSeries("SMA", myStrategy.getSMA())
    plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
    plt.getOrCreateSubplot("returns2").addDataSeries("CumulativeReturns", returnsAnalyzer.getCumulativeReturns())

    def parameters_generator():
        instrument = ["dia"]
        entrySMA = range(150, 251)
        exitSMA = range(5, 16)
        rsiPeriod = range(2, 11)
        overBoughtThreshold = range(75, 96)
        overSoldThreshold = range(5, 26)
        return itertools.product(instrument, entrySMA, exitSMA, rsiPeriod, overBoughtThreshold, overSoldThreshold)

    # 6. 策略运行
    myStrategy.run()
    myStrategy.info("Final portfolio value: $%.2f" % myStrategy.getResult())
    myStrategy.info("Final portfolio value: $%.2f" % myStrategy.getBroker().getEquity())
    plt.plot()


if __name__ == '__main__':
    # 策略：Strategies
    # 回测数据：Feeds
    # 交易经纪人：Brokers
    # 时间序列数据：DataSeries
    # 技术分析：Technicals
    # 优化器：Optimizer
    runStrategy()
