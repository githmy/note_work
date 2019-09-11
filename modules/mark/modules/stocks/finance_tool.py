import sys
import math
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.anova as anova
from statsmodels.formula.api import ols
import statsmodels.formula.api as sm
from scipy.stats import norm
from scipy import linalg
from statsmodels.tsa import stattools
from statsmodels.tsa import arima_model
from statsmodels.graphics.tsaplots import *
from arch.unitroot import ADF
from arch import arch_model
import ffn


# 指标元素生成
class ElementTool(object):
    """
    Sma: 移动平均  
    wma: 加权移动平均
    ema: 指数移动平均
    ewma: 指数加权移动平均
    OBV: On Balance Volume, 多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V
    """

    def __init__(self):
        returns = None
        self.returns = returns

    # 以下的为均方差
    def minVar(self, goalRet):
        covs = np.array(self.returns.cov())
        means = np.array(self.returns.mean())
        L1 = np.append(np.append(covs.swapaxes(0, 1), [means], 0),
                       [np.ones(len(means))], 0).swapaxes(0, 1)
        L2 = list(np.ones(len(means)))
        L2.extend([0, 0])
        L3 = list(means)
        L3.extend([0, 0])
        L4 = np.array([L2, L3])
        L = np.append(L1, L4, 0)
        results = linalg.solve(L, np.append(np.zeros(len(means)), [1, goalRet], 0))
        return np.array([list(self.returns.columns), results[:-2]])

    def frontierCurve(self):
        goals = [x / 500000 for x in range(-100, 4000)]
        variances = list(map(lambda x: self.calVar(self.minVar(x)[1, :].astype(np.float)), goals))
        plt.plot(variances, goals)

    def meanRet(self, fracs):
        meanRisky = ffn.to_returns(self.returns).mean()
        assert len(meanRisky) == len(fracs), 'Length of fractions must be equal to number of assets'
        return np.sum(np.multiply(meanRisky, np.array(fracs)))

    def calVar(self, fracs):
        return np.dot(np.dot(fracs, self.returns.cov()), fracs)

    # 以下的为移动平均
    def smaCal(self, tsPrice, k):
        # Sma = pd.Series(0.0, index=tsPrice.index)
        Sma = pd.Series(index=tsPrice.index)
        for i in range(k - 1, len(tsPrice)):
            Sma[i + 1] = sum(tsPrice[(i - k + 1):(i + 1)]) / k
        return Sma

    def wmaCal(self, tsPrice, weight):
        k = len(weight)
        arrWeight = np.array(weight)
        # Wma = pd.Series(0.0, index=tsPrice.index)
        Wma = pd.Series(index=tsPrice.index)
        for i in range(k - 1, len(tsPrice.index)):
            Wma[i + 1] = sum(arrWeight * tsPrice[(i - k + 1):(i + 1)])
        return Wma

    def emaCal(self, tsprice, period=5, exponential=0.2):
        Ema = pd.Series(0.0, index=tsprice.index)
        Ema[period - 1] = np.mean(tsprice[:period])
        for i in range(period + 1, len(tsprice)):
            expo = np.array(sorted(range(i - period + 1), reverse=True))
            w = (1 - exponential) ** expo
            Ema[i] = sum(exponential * w * tsprice[period:(i + 1)]) + tsprice * exponential ** (i - period)
        return Ema

    def ewmaCal(self, tsprice, period=5, exponential=0.2):
        Ewma = pd.Series(0.0, index=tsprice.index)
        Ewma[period - 1] = np.mean(tsprice[0:period])
        for i in range(period, len(tsprice)):
            Ewma[i] = exponential * tsprice[i] + (1 - exponential) * Ewma[i - 1]
        return Ewma

    # 动量
    def momentum(self, price, periond):
        lagPrice = price.shift(periond)
        momen = price - lagPrice
        momen = momen.dropna()
        return momen

    # 涨幅
    def rise_n(self, price, periond=1):
        prePrice = price.shift(-periond)
        return price / prePrice
        # return (price - prePrice) / prePrice

    # 预涨跌std 周期区段内 相对于区段的第1日 (类似布林带)
    def pre_up_down_std(self, tsPrice, period=20):
        # 半方差公式 只算下降的
        def cal_up_half_dev(relative_value):
            tmp = relative_value[relative_value > 1.0]
            if len(tmp) > 0:
                half_pre_up_std = (sum((tmp - 1.0) ** 2) / len(tmp)) ** 0.5
            else:
                half_pre_up_std = None
            return half_pre_up_std

        def cal_down_half_dev(relative_value):
            tmp = relative_value[relative_value <= 1.0]
            if len(tmp) > 0:
                half_pre_down_std = (sum((1.0 - tmp) ** 2) / len(tmp)) ** 0.5
            else:
                half_pre_down_std = None
            return half_pre_down_std

        # 初始值不可能为空
        pre_up_BBand_std = pd.Series(None, index=tsPrice.index)
        pre_down_BBand_std = pd.Series(None, index=tsPrice.index)
        for i in range(period, len(tsPrice) + 1):
            tmp_peri = tsPrice[i - period:i] / tsPrice[i - period + 1]
            # 半方差的大小
            pre_up_BBand_std[i] = cal_up_half_dev(tmp_peri)
            pre_down_BBand_std[i] = cal_down_half_dev(tmp_peri)
        return pre_up_BBand_std, pre_down_BBand_std

    # 最高低幅值
    def max_highlow_ret_aft_n(self, priceall, period=1):
        highret = pd.Series(index=priceall["close"].index)
        lowret = pd.Series(index=priceall["close"].index)
        for i in range(1, len(lowret) - period):
            highret[i] = max(priceall["high"][i - 1:i + period - 1]) / priceall["close"][i]
            lowret[i] = min(priceall["low"][i - 1:i + period - 1]) / priceall["close"][i]
        return highret, lowret

    # 最大涨跌
    def max_fallret_raiseret_aft_n(self, price, period=20):
        maxfallret = pd.Series(index=price.index)
        maxraiseret = pd.Series(index=price.index)
        for i in range(1, len(price) - period):
            tmpsec = price[i - 1:i + period - 1]
            tmpmax = price[i]
            tmpmin = price[i]
            tmpdrawdown = [1.0]
            tmpdrawup = [1.0]
            for t in range(i, i + period):
                if tmpsec[t] > tmpmax:
                    tmpmax = tmpsec[t]
                    tmpdrawdown.append(tmpdrawdown[-1])
                    tmpdrawup.append((tmpmax - tmpmin) / tmpmin)
                elif tmpsec[t] <= tmpmin:
                    tmpmin = tmpsec[t]
                    tmpdrawup.append(tmpdrawup[-1])
                    tmpdrawdown.append((tmpmax - tmpmin) / tmpmax)
                else:
                    pass
            maxfallret[i] = min(tmpdrawdown)
            maxraiseret[i] = max(tmpdrawup)
        return maxraiseret, maxfallret

    # 11. 相对强弱指数RSI RSI:= SMA(MAX(Close-LastClose,0),N,1)/SMA(ABS(Close-LastClose),N,1)*100
    def rsi(self, price, period=6):
        clprcChange = price - price.shift(1)
        clprcChange = clprcChange.dropna()
        indexprc = clprcChange.index
        upPrc = pd.Series(0, index=indexprc)
        upPrc[clprcChange > 0] = clprcChange[clprcChange > 0]
        downPrc = pd.Series(0, index=indexprc)
        downPrc[clprcChange < 0] = -clprcChange[clprcChange < 0]
        rsidata = pd.concat([price, clprcChange, upPrc, downPrc], axis=1)
        rsidata.columns = ['price', 'PrcChange', 'upPrc', 'downPrc']
        rsidata = rsidata.dropna()
        SMUP = []
        SMDOWN = []
        for i in range(period, len(upPrc) + 1):
            SMUP.append(np.mean(upPrc.values[(i - period):i], dtype=np.float32))
            SMDOWN.append(np.mean(downPrc.values[(i - period):i], dtype=np.float32))
            rsi = [100 * SMUP[i] / (SMUP[i] + SMDOWN[i]) for i in range(0, len(SMUP))]
        indexRsi = indexprc[(period - 1):]
        rsi = pd.Series(rsi, index=indexRsi)
        return rsi

    # BBands
    def bbands(self, tsPrice, period=20, times=2):
        upBBand = pd.Series(0.0, index=tsPrice.index)
        midBBand = pd.Series(0.0, index=tsPrice.index)
        downBBand = pd.Series(0.0, index=tsPrice.index)
        sigma = pd.Series(0.0, index=tsPrice.index)
        for i in range(period - 1, len(tsPrice)):
            midBBand[i] = np.nanmean(tsPrice[i - (period - 1):(i + 1)])
            sigma[i] = np.nanstd(tsPrice[i - (period - 1):(i + 1)])
            upBBand[i] = midBBand[i] + times * sigma[i]
            downBBand[i] = midBBand[i] - times * sigma[i]
        BBands = pd.DataFrame({'upBBand': upBBand[(period - 1):],
                               'midBBand': midBBand[(period - 1):],
                               'downBBand': downBBand[(period - 1):],
                               'sigma': sigma[(period - 1):]})
        return BBands

    # 16. 上下突破
    def upbreak(self, Line, RefLine):
        signal = np.all([Line > RefLine, Line.shift(1) < RefLine.shift(1)], axis=0)
        return pd.Series(signal[1:], index=Line.index[1:])

    def downbreak(self, Line, RefLine):
        signal = np.all([Line < RefLine, Line.shift(1) > RefLine.shift(1)], axis=0)
        return pd.Series(signal[1:], index=Line.index[1:])

    # 17. 成交量指标
    def VOblock(self, vol):
        return [np.sum(vol[BreakClose == x]) for x in range(6, 22, 2)]


# 交易策略生成
class TradeTool(object):
    # 以下的为 PairTrading 函数
    def SSD(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        returnX = (priceX - priceX.shift(1)) / priceX.shift(1)[1:]
        returnY = (priceY - priceY.shift(1)) / priceY.shift(1)[1:]
        standardX = (returnX + 1).cumprod()
        standardY = (returnY + 1).cumprod()
        SSD = np.sum((standardY - standardX) ** 2)
        return SSD

    def SSDSpread(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        retx = priceX.diff()[1:]
        rety = priceY.diff()[1:]
        standardX = (1 + retx).cumprod()
        standardY = (1 + rety).cumprod()
        spread = standardY - standardX
        return spread

    def cointegration(self, priceX, priceY):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        priceX = np.log(priceX)
        priceY = np.log(priceY)
        results = sm.OLS(priceY, sm.add_constant(priceX)).fit()
        resid = results.resid
        adfSpread = ADF(resid)
        if adfSpread.pvalue >= 0.05:
            print('''交易价格不具有协整关系.
            P-value of ADF test: %f
            Coefficients of regression:
            Intercept: %f
            Beta: %f
             ''' % (adfSpread.pvalue, results.params[0], results.params[1]))
            return None
        else:
            print('''交易价格具有协整关系.
            P-value of ADF test: %f
            Coefficients of regression:
            Intercept: %f
            Beta: %f
             ''' % (adfSpread.pvalue, results.params[0], results.params[1]))
            return results.params[0], results.params[1]

    def CointegrationSpread(self, priceX, priceY, formPeriod, tradePeriod):
        if priceX is None or priceY is None:
            print('缺少价格序列.')
        if not (re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', formPeriod)
                or re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', tradePeriod)):
            print('形成期或交易期格式错误.')
        formX = priceX[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        formY = priceY[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        coefficients = self.cointegration(formX, formY)
        if coefficients is None:
            print('未形成协整关系,无法配对.')
        else:
            spread = (np.log(priceY[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]])
                      - coefficients[0] - coefficients[1] * np.log(
                priceX[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]))
            return spread

    def calBound(self, priceX, priceY, method, formPeriod, width=1.5):
        if not (re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', formPeriod)
                or re.fullmatch('\d{4}-\d{2}-\d{2}:\d{4}-\d{2}-\d{2}', tradePeriod)):
            print('形成期或交易期格式错误.')
        if method == 'SSD':
            spread = self.SSDSpread(priceX[formPeriod.split(':')[0]:formPeriod.split(':')[1]],
                                    priceY[formPeriod.split(':')[0]:formPeriod.split(':')[1]])
            mu = np.mean(spread)
            sd = np.std(spread)
            UpperBound = mu + width * sd
            LowerBound = mu - width * sd
            return UpperBound, LowerBound
        elif method == 'Cointegration':
            spread = self.CointegrationSpread(priceX, priceY, formPeriod, formPeriod)
            mu = np.mean(spread)
            sd = np.std(spread)
            UpperBound = mu + width * sd
            LowerBound = mu - width * sd
            return UpperBound, LowerBound
        else:
            print('不存在该方法. 请选择"SSD"或是"Cointegration".')

    def TradeSimPair(self, priceX, priceY, position):
        n = len(position)
        size = 1000
        shareY = size * position
        shareX = [(-beta) * shareY[0] * priceY[0] / priceX[0]]
        cash = [2000]
        for i in range(1, n):
            shareX.append(shareX[i - 1])
            cash.append(cash[i - 1])
            if position[i - 1] == 0 and position[i] == 1:
                shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
                cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
            elif position[i - 1] == 0 and position[i] == -1:
                shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
                cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
            elif position[i - 1] == 1 and position[i] == 0:
                shareX[i] = 0
                cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
            elif position[i - 1] == -1 and position[i] == 0:
                shareX[i] = 0
                cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
        cash = pd.Series(cash, index=position.index)
        shareY = pd.Series(shareY, index=position.index)
        shareX = pd.Series(shareX, index=position.index)
        asset = cash + shareY * priceY + shareX * priceX
        account = pd.DataFrame(
            {'Position': position, 'ShareY': shareY, 'ShareX': shareX, 'Cash': cash, 'Asset': asset})
        return account

    # 19. 单交易
    def TradeSim(self, price, hold):
        position = pd.Series(np.zeros(len(price)), index=price.index)
        position[hold.index] = hold.values
        cash = 20000 * np.ones(len(price))
        for t in range(1, len(price)):
            if position[t - 1] == 0 and position[t] > 0:
                cash[t] = cash[t - 1] - price[t] * 1000
            if position[t - 1] >= 1 and position[t] == 0:
                cash[t] = cash[t - 1] + price[t] * 1000
            if position[t - 1] == position[t]:
                cash[t] = cash[t - 1]
        asset = cash + price * position * 1000
        asset.name = 'asset'
        account = pd.DataFrame({'asset': asset, 'cash': cash, 'position': position})
        return account

    def makviz(self):
        """
        效用U(simple_return)
        组分权重w
        E(U(sigma(w*simple_return)))  s.t. sigma(w)=1 
        min σ^2(simple_return)=sigma(wi^2*σi(simple_return)^2)+sigma(wi*wj*σ(simple_return i,simple_return j)) s.t. simple_return.mean=sigma(wi*E(simple_return))  
        :return: 
        """
        close = pd.DataFrame()["close"]
        # 相关性协方差 series (列2) ， DataFrame (空) 返回矩阵
        close.corr()

    # blminVar
    def blminVar(self, blres, goalRet):
        covs = np.array(blres[1])
        means = np.array(blres[0])
        L1 = np.append(np.append((covs.swapaxes(0, 1)), [means.flatten()], 0),
                       [np.ones(len(means))], 0).swapaxes(0, 1)
        L2 = list(np.ones(len(means)))
        L2.extend([0, 0])
        L3 = list(means)
        L3.extend([0, 0])
        L4 = np.array([L2, L3])
        L = np.append(L1, L4, 0)
        results = linalg.solve(L, np.append(np.zeros(len(means)), [1, goalRet], 0))
        return pd.DataFrame(results[:-2], index=blres[1].columns, columns=['p_weight'])

    def conponent_profit(self):
        close = pd.DataFrame()["close"]
        returns = ffn.get('aapl,msft,c,gs,ge', start='2010-01-01').to_returns(close).dropna()
        returns.calc_mean_var_weights().as_format('.2%')
        # 1. 方法
        Tesla['Return'] = (Tesla['Close'] - Tesla['Close'].shift(1)) / Tesla['Close'].shift(1)
        Tesla = Tesla.dropna()
        # 2. 方法
        GM['Return'] = ffn.to_returns(GM['Close'])
        # 3. 方法
        Ford['Return'] = Ford['Close'].pct_change(1)
        Ford = Ford.dropna()
        # 年化率
        simpleret = ffn.to_returns(close)
        # 复利化
        simpleret = ffn.to_log_returns(close)
        annue = (1 + simpleret).cumprod()[-1] ** (245 / 311) - 1
        # 方差
        simpleret.std()

    def blacklitterman(self, returns, tau, P, Q):
        mu = returns.mean()
        sigma = returns.cov()
        pi1 = mu
        ts = tau * sigma
        Omega = np.dot(np.dot(P, ts), P.T) * np.eye(Q.shape[0])
        middle = linalg.inv(np.dot(np.dot(P, ts), P.T) + Omega)
        er = np.expand_dims(pi1, axis=0).T + np.dot(np.dot(np.dot(ts, P.T), middle),
                                                    (Q - np.expand_dims(np.dot(P, pi1.T), axis=1)))
        posteriorSigma = sigma + ts - np.dot(ts.dot(P.T).dot(middle).dot(P), ts)
        return [er, posteriorSigma]

    # 时间T内，组合损失X的置信度小于1-a%
    def value_at_risk(self):
        # 半方差公式 只算下降的
        def cal_half_dev(returns):
            # 均值
            mu = returns.mean()
            tmp = returns["returnss" < mu]
            half_deviation = (sum((mu - tmp) ** 2) / len(returns)) ** 0.5
            return half_deviation

        close = pd.DataFrame()["close"]
        simple_return = ffn.to_returns(close)
        # 半方差的大小
        res = cal_half_dev(simple_return)
        print(res)
        # 历史模拟
        simple_return.quantile(0.05)
        # 风险值最差0.05的位置
        norm.ppf(0.05, simple_return.mean(), simple_return.std())
        # 最差0.05的均值期望
        worst_return = simple_return["returnss" < simple_return.quantile(0.05)].mean()
        # 最大回撤 展开
        price = (1 + simple_return).cumprod()  # 返回的依然是数组
        simple_return.cummax() - price
        # 最大回撤
        ffn.calc_max_drawdown(price)
        ffn.calc_max_drawdown((1 + simple_return).cumprod())

    # 14. 布林带风险
    def CalBollRisk(self, tsPrice, multiplier):
        et = ElementTool()
        k = len(multiplier)
        overUp = []
        belowDown = []
        BollRisk = []
        for i in range(k):
            BBands = et.bbands(tsPrice, 20, multiplier[i])
            a = 0
            b = 0
            for j in range(len(BBands)):
                tsPrice = tsPrice[-(len(BBands)):]
                if tsPrice[j] > BBands.upBBand[j]:
                    a += 1
                elif tsPrice[j] < BBands.downBBand[j]:
                    b += 1
            overUp.append(a)
            belowDown.append(b)
            BollRisk.append(100 * (a + b) / len(tsPrice))
        return BollRisk

    # 策略函数
    def strategy_analy(self, tradeSignal, ret):
        indexDate = tradeSignal.index
        ret = ret[indexDate]
        tradeRet = ret * tradeSignal
        tradeRet[tradeRet == (-0)] = 0
        winRate = len(tradeRet[tradeRet > 0]) / len(tradeRet[tradeRet != 0])
        meanWin = sum(tradeRet[tradeRet > 0]) / len(tradeRet[tradeRet > 0])
        meanLoss = sum(tradeRet[tradeRet < 0]) / len(tradeRet[tradeRet < 0])
        perform = {'winRate': winRate, 'meanWin': meanWin, 'meanLoss': meanLoss}
        return perform

    # 18. 判断持有
    def judge_hold(self, signal):
        hold = np.zeros(len(signal))
        for index in range(1, len(hold)):
            if hold[index - 1] == 0 and signal[index] == 1:
                hold[index] = 1
            elif hold[index - 1] == 1 and signal[index] == 1:
                hold[index] = 1
            elif hold[index - 1] == 1 and signal[index] == 0:
                hold[index] = 1
        return pd.Series(hold, index=signal.index)

    # 19. 凯里公式
    def kari_normal_w(self, p, q, fw, fl):
        c = p * fw - q * fl
        a = fw * fl
        wm = c / a
        return wm

    # 19. 凯里公式
    def kari_normal_g(self, p, q, fw, fl, w):
        g = np.power((1 + w * fw), p) * np.power((1 - w * fl), q)
        return g

    # 20. 凯里公式修正
    def kari_fix_normal_w(self, p, q, fw, fl, system_risk):
        a = fw * fl
        b = -(1 + fl) * fw * p + (1 - fw) * fl * q + (fl - fw) * system_risk
        c = p * fw - q * fl - system_risk
        wm = (-b - np.sqrt(np.square(b) - 4 * a * c)) / (2 * a)
        return wm

    # 20. 凯里公式修正
    def kari_fix_normal_g(self, p, q, fw, fl, system_risk, w):
        g = np.power((1 + w * fw), p) * np.power((1 - w * fl), q) * np.power((1 - w), system_risk)
        return g


# 回测生成
class BackTestTool(object):
    def performance(self, x):
        winpct = len(x[x > 0]) / len(x[x != 0])
        annRet = (1 + x).cumprod()[-1] ** (245 / len(x)) - 1
        sharpe = ffn.calc_risk_return_ratio(x)
        maxDD = ffn.calc_max_drawdown((1 + x).cumprod())
        perfo = pd.Series([winpct, annRet, sharpe, maxDD],
                          index=['win rate', 'annualized return', 'sharpe ratio', 'maximum drawdown'])
        return perfo

    # 性能
    def perform(self, tsPrice, tsTradSig):
        ret = tsPrice / tsPrice.shift(1) - 1
        tradRet = (ret * tsTradSig).dropna()
        ret = ret[-len(tradRet):]
        winRate = [len(ret[ret > 0]) / len(ret[ret != 0]),
                   len(tradRet[tradRet > 0]) / len(tradRet[tradRet != 0])]
        meanWin = [np.mean(ret[ret > 0]),
                   np.mean(tradRet[tradRet > 0])]
        meanLoss = [np.mean(ret[ret < 0]),
                    np.mean(tradRet[tradRet < 0])]
        Performance = pd.DataFrame({'winRate': winRate, 'meanWin': meanWin,
                                    'meanLoss': meanLoss})
        Performance.index = ['Stock', 'Trade']
        return Performance

    # 15. 交易
    def trade(self, signal, price):
        ret = ((price - price.shift(1)) / price.shift(1))[1:]
        ret.name = 'ret'
        signal = signal.shift(1)[1:]
        tradeRet = ret * signal + 0
        tradeRet.name = 'tradeRet'
        Returns = pd.merge(pd.DataFrame(ret), pd.DataFrame(tradeRet),
                           left_index=True, right_index=True).dropna()
        return Returns

    def backtest(self, ret, tradeRet):
        BuyAndHold = self.performance(ret)
        Trade = self.performance(tradeRet)
        return pd.DataFrame({ret.name: BuyAndHold, tradeRet.name: Trade})


class FitTool(object):
    def liner_demo(self):
        pd_data = pd.DataFrame()
        model = sm.OLS(np.log(pd_data["depend_var"]),
                       sm.add_constant(pd_data[["constant_column1", "constant_column2"]])).fit()
        print(model.summary())
        # pvalue 小于0.05的可以作为系数  y = coef * log(depend_var) + coef * constant_columns

    def ols_model(self):
        # 1. 单因素方差分析
        # 常规最小方差
        pddata = pd.read_csv()
        shindex = pddata[pddata.indexcd == 1]
        szindex = pddata[pddata.indexcd == 399106]
        model = ols('因变量 ~C(自变量)', data=pddata.dropna()).fit()
        table1 = anova.anova_lm(model)
        # p值小于0.05 说明意外少，既同步相关
        print(table1)
        # 2. 多因素方差分析
        model = ols('因变量 ~ C(削弱的自变量)+C(增强的自变量)', data=pddata.dropna()).fit()
        table2 = anova.anova_lm(model)
        # p值小于0.05 说明意外少，既同步相关
        # 3. 析因方差分析
        model = ols('因变量 ~ C(自变量1)*C(自变量2)', data=pddata.dropna()).fit()
        table3 = anova.anova_lm(model)
        # p值大于0.05 说明不是偶然，既不同步，不相关
        # 拟合的值，残差,pearson残差
        print(model.fittedvalues, model.resid, model.resid_pearson)


def many_paras():
    """
    p-value本质是控制假阳性率（False positive rate，FPR）
    q-value 控制的是FDR (false discovery rate)
    Q-statistic: Qlb=T*(T+2)*sigma(j=1,p)(rj^2/(T-j))  rj残差序列，j阶自相关系数，T观测值的个数，p滞后阶数。
    FDR = E（V/R） 错误发现次数V，总的拒绝次数R
    acf: 自相关系数 -- y(t)= a0 + a1*y(t-1) + epsilon
         p(x(i)|x(i-h)) :sigma(i=1,n-h) ((x(i)-mu)*(x(i+h)-mu)/sigma(i=1,n) ((x(i)-mu)^2))
    pacf: 偏自相关系数，k-1个时间滞后 作为已知，只求k -- y(t)= a0 + a1*y(t-1) + ... a1*y(t-k) + epsilon
         p(x(i)..x(i-k)|x(i-1)x(i-k+1)) :
    ARMA(p，q): AR代表p阶自回归过程，MA代表q阶移动平均过程
    ARIMA模型是在ARMA模型的基础上多了差分的操作。
    ADF: 白噪声随机干扰项的一阶自回归过程。用单位根 检验，存在就是非平稳。y(t)= mu + fi*y(t-1) + epsilon。p阶要求 p个根的和小于1。
    Sma: 移动平均  
    wma: 加权移动平均
    ema: 指数移动平均
    ewma: 指数加权移动平均
    OBV: On Balance Volume, 多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V
    :return: 
    """
    # 1. 计算自相关系数
    acfs = stattools.acf(SHRet)
    # 绘制自相关系数图
    plot_acf(SHRet, use_vlines=True, lags=30)
    # 2. 计算偏自相关系数
    pacfs = stattools.pacf(SHRet)
    plot_pacf(SHRet, use_vlines=True, lags=30)
    # 3. 进行ADF单位根检验，并查看结果；
    adfSHRet = ADF(SHRet)
    print(adfSHRet.summary().as_text())
    # 4. Q 统计
    LjungBox1 = stattools.q_stat(stattools.acf(SHRet)[1:13], len(SHRet))
    print(LjungBox1)
    # 5. lag即为上述检验表达式中的m，在这里我们选择检验12阶的自相关系数。
    LjungBox = stattools.q_stat(stattools.acf(CPItrain)[1:12], len(CPItrain))
    # order表示建立的模型的阶数，c(1,0,1)表示建立的是ARMA(1,1)模型；
    # 中间的数字0表示使用原始的、未进行过差分（差分次数为0）的数据；
    model1 = arima_model.ARIMA(CPItrain, order=(1, 0, 1)).fit()
    model1.summary()
    model1.conf_int()
    # 6. 绘制时间序列模拟的诊断图
    stdresid = model1.resid / math.sqrt(model1.sigma2)
    plt.plot(stdresid)
    plot_acf(stdresid, lags=20)
    LjungBox = stattools.q_stat(stattools.acf(stdresid)[1:13], len(stdresid))
    print(LjungBox[1][-1])
    print(model1.forecast(3)[0])
    # 7. Autoregressive conditional heteroskedasticity model 自回归条件异方差模型
    # y(t)=b*x(t)+epsilon(t)
    # epsilon(t)^2=a0+a1*epsilon(t-1)^2+a2*epsilon(t-2)^2+n(t)
    # \sigma_t^{2}=\omega+\sum_{i=1}^{p}\alpha_{i}\epsilon_{t-i}^{2}
    # n(t)独立同分布 期望为0，var(n^2)=r^2
    am = arch_model(SHret)
    model = am.fit(update_freq=0)
    print(model.summary())
    # 8. 对子 的 处理
    pt = TradeTool()
    SSD = pt.SSD(priceAf, priceBf)
    SSDspread = pt.SSDSpread(priceAf, priceBf)
    SSDspread.describe()
    coefficients = pt.cointegration(priceAf, priceBf)
    CoSpreadF = pt.CointegrationSpread(priceA, priceB, formPeriod, formPeriod)
    CoSpreadTr = pt.CointegrationSpread(priceA, priceB, formPeriod, tradePeriod)
    CoSpreadTr.describe()
    bound = pt.calBound(priceA, priceB, 'Cointegration', formPeriod, width=1.2)
    # 9. 配对 选点
    trtl = TradeTool()
    account = trtl.TradeSimPair(PAt, PBt, position)

    # 10. momentum function
    et = ElementTool()
    et.momentum(Close, 5).tail(n=5)
    momen35 = et.momentum(Close, 35)
    signal = []
    for i in momen35:
        if i > 0:
            signal.append(1)
        else:
            signal.append(-1)
    signal = pd.Series(signal, index=momen35.index)
    signal.head()
    tradeSig = signal.shift(1)
    ret = Close / Close.shift(1) - 1
    # ret=ret['2014-02-20':]
    # ret.head(n=3)
    Mom35Ret = ret * (signal.shift(1))
    Mom35Ret[0:5]
    real_Mom35Ret = Mom35Ret[Mom35Ret != 0]
    real_ret = ret[ret != 0]

    Rsi12 = et.rsi(BOCMclp, 12)
    # 策略
    rsi6 = et.rsi(BOCMclp, 6)
    rsi24 = et.rsi(BOCMclp, 24)
    # rsi6捕捉买卖点
    Sig1 = []
    for i in rsi6:
        if i > 80:
            Sig1.append(-1)
        elif i < 20:
            Sig1.append(1)
        else:
            Sig1.append(0)

    date1 = rsi6.index
    Signal1 = pd.Series(Sig1, index=date1)
    Signal1[Signal1 == 1].head(n=3)
    Signal1[Signal1 == -1].head(n=3)

    Signal2 = pd.Series(0, index=rsi24.index)
    lagrsi6 = rsi6.shift(1)
    lagrsi24 = rsi24.shift(1)
    for i in rsi24.index:
        if (rsi6[i] > rsi24[i]) & (lagrsi6[i] < lagrsi24[i]):
            Signal2[i] = 1
        elif (rsi6[i] < rsi24[i]) & (lagrsi6[i] > lagrsi24[i]):
            Signal2[i] = -1

    signal = Signal1 + Signal2
    signal[signal >= 1] = 1
    signal[signal <= -1] = -1
    signal = signal.dropna()
    tradSig = signal.shift(1)

    tt = TradeTool()
    BuyOnly = tt.strategy_analy(buy, ret)
    SellOnly = tt.strategy_analy(sell, ret)
    Trade = tt.strategy_analy(tradSig, ret)
    Test = pd.DataFrame({"BuyOnly": BuyOnly, "SellOnly": SellOnly, "Trade": Trade})

    # 累计收益率
    cumStock = np.cumprod(1 + ret) - 1
    cumTrade = np.cumprod(1 + tradeRet) - 1

    # 12. 移动平均线
    sma5 = et.smaCal(Close, 5)
    # 12. 加权移动平均线
    wma5 = et.wmaCal(Close, w)
    # 12. 指数移动平均线
    Ema = et.emaCal(Close, period)
    print(Ema)
    # 12. 指数加权移动平均线
    Ewma = et.ewmaCal(Close, 5, 0.2)

    # 13. 布林带
    UnicomBBands = et.bbands(Close, 20, 2)
    print(UnicomBBands)
    multiplier = [1, 1.65, 1.96, 2, 2.58]
    price2010 = Close['2010-01-04':'2010-12-31']
    tt.CalBollRisk(price2010, multiplier)

    # 14. 性能
    btt = BackTestTool()
    Performance1 = btt.perform(Close, tradSignal1)
    print(Performance1)

    # 15. 交易, 回测
    KDtrade = btt.trade(KDSignal, close)
    btt.backtest(KDtrade.Ret, KDtrade.KDtradeRet)

    # 16. 上下突破
    KDupbreak = et.upbreak(KValue, DValue) * 1
    KDupbreak[KDupbreak == 1].head()

    KDdownbreak = et.downbreak(KValue, DValue) * 1
    KDdownbreak[KDdownbreak == 1].head()
    # "金叉"与"死叉"交易策略绩效表现
    btt.backtest(KDbreak.Ret, KDbreak.KDbreakRet)

    # 17. 成交量指标
    cumUpVol = et.VOblock(UpVol)
    cumDownVol = et.VOblock(DownVol)
    ALLVol = np.array([cumUpVol, cumDownVol]).transpose()

    # 18. 判断持有
    hold = tt.judge_hold(trade)

    # 19. 单交易
    TradeAccount = tt.TradeSim(close, hold)
    print(TradeAccount)


def main(args=None):
    et = ElementTool(sh_return)
    et.frontierCurve()
    goal_return = 0.003
    et.minVar(goal_return)
    P = np.array([pick1, pick2])
    Q = np.array([q1, q2])
    tt = TradeTool()
    res = tt.blacklitterman(sh_return, 0.1, P, Q)
    tt.blminVar(res, 0.75 / 252)
    # 1. 命令行
    # parajson = get_paras(args)


if __name__ == '__main__':
    # 1. 参数解析
    main(sys.argv[1:])
