# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import os
import numpy as np
import logging
import tushare as ts
import datetime
import re
import matplotlib.pyplot as plt
from itertools import combinations
from modules.stocks.stock_data import LocalStockdata


# 生成序列特征
class Sequence_chara(object):
    def __init__(self, pf, parajson):
        # 0. 基本元素
        self.charamember = ['CCI', 'EVM', 'SMA', 'BBup', 'BBdown', 'EWMA', 'FI', 'ROC']
        # 1. 数据
        self.pf = pf
        # 2. 默认参数
        if parajson is None:
            self.parajson = {
                "avenlist": [],
                "labellist": [1],
            }
            # self.parajson = {
            #     "avenlist": [5, 20, 60],
            #     "labellist": [1, 2, 4, 8, 16, 32, 64, 128, 256],
            # }
        else:
            self.parajson = parajson
        # 3. 基本曲线
        self.chara_volume_diff()
        for i1 in self.parajson["avenlist"]:
            self.get_basic_chara_n(ndays=i1)
        # 4. 标记点位
        self.get_cross_chara()
        # 5. 标签
        self.get_label_avelist(self.parajson["labellist"])
        # 6. 学习
        # 7. 验证
        # 8. 对冲荷兰赌
        # 9. 强化
        # 10. 交易框架

    def get_label_avelist(self, lists):
        if lists is not None:
            for i1 in lists:
                lup = (self.pf['high'].rolling(window=i1).max() - self.pf['close'].shift(i1)) / self.pf['close'].shift(
                    i1)
                ldw = (self.pf['low'].rolling(window=i1).max() - self.pf['close'].shift(i1)) / self.pf['close'].shift(
                    i1)
                lcu = (self.pf['close'] - self.pf['close'].shift(i1)) / self.pf['close'].shift(i1)
                lb = pd.Series(lup.shift(-i1), name='ylabel_' + str(i1) + "_u")
                lc = pd.Series(lcu.shift(-i1), name='ylabel_' + str(i1) + "_c")
                ld = pd.Series(ldw.shift(-i1), name='ylabel_' + str(i1) + "_d")
                self.pf = self.pf.join(lb)
                self.pf = self.pf.join(lc)
                self.pf = self.pf.join(ld)

    def get_cross_chara(self):
        combins = [c for c in combinations(map(str, self.parajson["avenlist"]), 2)]
        for i1 in self.pf.columns:
            if i1.startswith("chara_"):
                splist = i1.split("_")
                for i2 in combins:
                    # o：over 后面的线-前面的线
                    # x：cross + m
                    # s：sig +同侧没变 -异侧变化
                    oriname1 = "_".join([splist[0], i2[0], splist[2]])
                    oriname2 = "_".join([splist[0], i2[1], splist[2]])
                    over2_1 = "_".join(["charao", i2[0], i2[1], splist[2]])
                    sig2_1 = "_".join(["charas", i2[0], i2[1], splist[2]])
                    # cross2_1 = "_".join(["charax", i2[0], i2[1], splist[2]])
                    # self.pf.eval(over2_1 + "=" + oriname2 + "-" + oriname1, inplace=True)
                    self.pf[over2_1] = self.pf[oriname2] - self.pf[oriname1]
                    # self.pf["tmp2"] = self.pf[over2_1].shift(-1)
                    # self.pf.eval(newcolname + "=" + "tmp2" + "-" + over2_1, inplace=True)
                    self.pf[sig2_1] = np.sign(self.pf[over2_1].shift(-1) * self.pf[over2_1])
                    # self.pf[cross2_1] = np.sign(self.pf[over2_1].shift(-1) * self.pf[over2_1])
                    # self.pf[newcolname] = np.sign(self.pf[newcolname])

    def get_basic_chara_n(self, ndays=5):
        # 随顺市势指标 : CCI（N日）=（TP－MA）÷Std÷0.015
        self.chara_CCI(ndays)
        # 简易波动指标（Ease of Movement Value）又称EMV指标
        # EVM=（Current High Price - Current Low Price）/2 - （Prior High Price - Prior Low Price）/2
        self.chara_EVM(ndays)
        self.charaR_EVM(ndays)
        # 移动平均线指标MA 简单: N日MA=N日收市价的总和/N(即算术平均数)
        self.chara_SMA(ndays)
        self.charaR_SMA(ndays)
        # 移动平均线指标MA 指数权重: N日MA=N日收市价的总和/N(即算术平均数)
        self.chara_EWMA(ndays)
        self.charaR_EWMA(ndays)
        # 布林线指标（BB）: 统计原理，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        # 中轨线=N日的移动平均线
        # 上轨线=中轨线+两倍的标准差
        # 下轨线=中轨线－两倍的标准差
        self.chara_BB(ndays)
        self.charaR_STDR(ndays)
        # 强力指数指标（Force Index）: 上升或下降趋势的力量大小，在零线上下移动来表示趋势的强弱。
        # FORCE INDEX（i）=VOLUME（i）*[MA（ApPRICE，N，i）-MA（ApPRICE，N，i-1）]
        # FORCE INDEX（i）：当前柱的力量指数
        # VOLUME（i）：当前柱的交易量；
        # MA（ApPRICE，N，i）：在任何一个时段内当前柱的任何移动平均线：
        # MA（ApPRICE，N，i-1）——前一柱的任何移动平均线。
        self.chara_FI(ndays)  # 归一化不方便，临时不用
        self.charaR_FI(ndays)
        # 变化速率ROC : 统计原理，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        self.chara_ROC(ndays)
        self.charaR_ROC(ndays)

    # 交易量 相对化
    def chara_volume_diff(self):
        ori = self.pf['volume'].diff(1) / self.pf['volume'].shift(1)
        orir = self.pf['volume'] / self.pf['volume'].shift(1)
        volori = pd.Series(ori, name='charaR_1_1_vol')
        volrori = pd.Series(orir, name='charaR_1_1_volr')
        volrlog = pd.Series(np.log(orir), name='charaR_1_1_volrlog')
        self.pf = self.pf.join(volori)
        self.pf = self.pf.join(volrori)
        self.pf = self.pf.join(volrlog)

    def chara_resample(self, ndays=5):
        # 更改样本的尺度 :
        if ndays is None:
            raise Exception("error: no ndays")
        df_ohlc = self.pf['close'].resample(ndays + 'D').ohlc()
        df_volume = self.pf['volume'].resample(ndays + 'D').sum()

    def chara_CCI(self, ndays=5):
        # 随顺市势指标 : CCI（N日）=（TP－MA）÷Std÷0.015
        if ndays is None:
            raise Exception("error: no ndays")
        TP = (self.pf['high'] + self.pf['low'] + self.pf['close']) / 3
        CCI = pd.Series((TP - TP.rolling(window=ndays).mean()) / (0.015 * TP.rolling(window=ndays).std()),
                        name='charaR_' + str(ndays) + '_CCI')
        self.pf = self.pf.join(CCI)

    def chara_EVM(self, ndays=5):
        # 简易波动指标（Ease of Movement Value）又称EMV指标
        # EVM=（Current High Price - Current Low Price）/2 - （Prior High Price - Prior Low Price）/2
        if ndays is None:
            raise Exception("error: no ndays")
        dm = ((self.pf['high'] + self.pf['low']) / 2) - ((self.pf['high'].shift(1) + self.pf['low'].shift(1)) / 2)
        br = (self.pf['volume'] / 100000000) / (self.pf['high'] - self.pf['low'])
        EVM = dm / br
        EVM_MA = pd.Series(EVM.rolling(window=ndays).mean(), name='chara_' + str(ndays) + '_EVM')
        self.pf = self.pf.join(EVM_MA)

    def charaR_EVM(self, ndays=5):
        # 简易波动指标（Ease of Movement Value）又称EMV指标
        # EVM=（Current High Price - Current Low Price）/2 - （Prior High Price - Prior Low Price）/2
        if ndays is None:
            raise Exception("error: no ndays")
        av = (self.pf['high'] + self.pf['low']) / 2
        dm = av - av.shift(1)
        dv = self.pf['high'] - self.pf['low']
        br = self.pf['charaR_1_1_vol'].abs() / dv
        EVM = dm * br
        EVM_MA = pd.Series(EVM.rolling(window=ndays).mean(), name='charaR_' + str(ndays) + '_EVM')
        self.pf = self.pf.join(EVM_MA)

    def chara_SMA(self, ndays=5):
        # 移动平均线指标MA 简单: N日MA=N日收市价的总和/N(即算术平均数)
        if ndays is None:
            raise Exception("error: no ndays")
        SMA = pd.Series(self.pf['close'].rolling(window=ndays).mean(), name='chara_' + str(ndays) + '_SMA')
        self.pf = self.pf.join(SMA)

    def charaR_SMA(self, ndays=5):
        # 移动平均线指标MA 简单: N日MA=N日收市价的总和/N(即算术平均数)
        if ndays is None:
            raise Exception("error: no ndays")
        SMA = pd.Series(self.pf['close'].rolling(window=ndays).mean() / self.pf['close'].shift(ndays),
                        name='charaR_' + str(ndays) + '_SMAR')
        self.pf = self.pf.join(SMA)

    def chara_EWMA(self, ndays=5):
        # 移动平均线指标MA 指数权重: N日MA=N日收市价的总和/N(即算术平均数)
        if ndays is None:
            raise Exception("error: no ndays")
        # EMA = pd.Series(pd.ewma(self.pf['close'], span=ndays, min_periods=ndays - 1),
        #                 name='chara_' + str(ndays) + '_EWMA')
        EMA = pd.Series(self.pf['close'].ewm(span=ndays).mean(), name='chara_' + str(ndays) + '_EWMA')
        self.pf = self.pf.join(EMA)

    def charaR_EWMA(self, ndays=5):
        # 移动平均线指标MA 指数权重: N日MA=N日收市价的总和/N(即算术平均数)
        if ndays is None:
            raise Exception("error: no ndays")
        EMA = pd.Series(self.pf['close'].ewm(span=ndays).mean() / self.pf['close'].shift(ndays),
                        name='charaR_' + str(ndays) + '_EWMAR')
        self.pf = self.pf.join(EMA)

    def chara_BB(self, ndays=5):
        # 布林线指标（BB）: 统计原理，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        # 中轨线=N日的移动平均线
        # 上轨线=中轨线+两倍的标准差
        # 下轨线=中轨线－两倍的标准差
        if ndays is None:
            raise Exception("error: no ndays")
        MA = pd.Series(self.pf['close'].rolling(window=ndays).mean())
        SD = pd.Series(self.pf['close'].rolling(window=ndays).std())
        b1 = MA + (2 * SD)
        B1 = pd.Series(b1, name='chara_' + str(ndays) + '_BBup')
        self.pf = self.pf.join(B1)
        b2 = MA - (2 * SD)
        B2 = pd.Series(b2, name='chara_' + str(ndays) + '_BBdown')
        self.pf = self.pf.join(B2)

    def charaR_STDR(self, ndays=5):
        # 布林线指标（BB）: 统计原理，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        # 中轨线=N日的移动平均线
        # 上轨线=中轨线+两倍的标准差
        # 下轨线=中轨线－两倍的标准差
        if ndays is None:
            raise Exception("error: no ndays")
        MA = pd.Series(self.pf['close'].rolling(window=ndays).mean())
        SD = pd.Series(self.pf['close'].rolling(window=ndays).std())
        B2 = pd.Series(SD / MA, name='charaR_' + str(ndays) + '_STDR')
        self.pf = self.pf.join(B2)

    def chara_FI(self, ndays=5):
        # 强力指数指标（Force Index）: 上升或下降趋势的力量大小，在零线上下移动来表示趋势的强弱。
        # FORCE INDEX（i）=VOLUME（i）*[MA（ApPRICE，N，i）-MA（ApPRICE，N，i-1）]
        # FORCE INDEX（i）：当前柱的力量指数
        # VOLUME（i）：当前柱的交易量；
        # MA（ApPRICE，N，i）：在任何一个时段内当前柱的任何移动平均线：
        # MA（ApPRICE，N，i-1）——前一柱的任何移动平均线。
        if ndays is None:
            raise Exception("error: no ndays")
        FI = pd.Series(self.pf['close'].diff(ndays) * self.pf['volume'], name='chara_' + str(ndays) + '_FI')
        self.pf = self.pf.join(FI)

    def charaR_FI(self, ndays=5):
        # 强力指数指标（Force Index）: 上升或下降趋势的力量大小，在零线上下移动来表示趋势的强弱。
        # FORCE INDEX（i）=VOLUME（i）*[MA（ApPRICE，N，i）-MA（ApPRICE，N，i-1）]
        # FORCE INDEX（i）：当前柱的力量指数
        # VOLUME（i）：当前柱的交易量；
        # MA（ApPRICE，N，i）：在任何一个时段内当前柱的任何移动平均线：
        # MA（ApPRICE，N，i-1）——前一柱的任何移动平均线。
        if ndays is None:
            raise Exception("error: no ndays")
        FI = pd.Series(self.pf['close'].diff(ndays) / self.pf['close'].shift(ndays) * np.exp(
            self.pf['charaR_1_1_volrlog'].rolling(window=ndays).sum()), name='charaR_' + str(ndays) + '_FI')
        self.pf = self.pf.join(FI)

    def chara_ROC(self, ndays=5):
        # 变化速率ROC : 统计原理，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        if ndays is None:
            raise Exception("error: no ndays")
        N = self.pf['close'].diff(ndays)
        D = self.pf['close'].shift(ndays)
        ROC = pd.Series(N / D, name='charaR_' + str(ndays) + '_ROC')
        self.pf = self.pf.join(ROC)

    def charaR_ROC(self, ndays=5):
        # 变化速率ROC : 统计原理，求出股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
        if ndays is None:
            raise Exception("error: no ndays")
        rela = self.pf['close'] / self.pf['close'].shift(ndays)
        ROCr = pd.Series(rela, name='charaR_' + str(ndays) + '_ROCr')
        ROCrlog = pd.Series(np.log(rela), name='charaR_' + str(ndays) + '_ROCrlog')
        self.pf = self.pf.join(ROCr)
        self.pf = self.pf.join(ROCrlog)


# 需求特征的组合
class Component_charas(object):
    def __init__(self):
        pass

    # 强化学习所需特征
    def reforce_charas(self, pdobj, parajson):
        # 可多可少，临时用。
        parajson = {
            "avenlist": [5, 20, 60],
            "labellist": [1, 2, 4, 8, 16],
        }
        dclass = Sequence_chara(pdobj, parajson)
        respd = dclass.pf
        # charalist = [i for i in respd.columns if not (i.startswith("ma") or i.startswith("v_ma"))]
        # respd = respd[charalist][respd["ylabel_1_c"].notnull()]
        respd = respd[respd.columns]
        del respd["turnover"]
        respd.dropna(inplace=True)
        return respd
        # charalist = [i for i in respd.columns if
        #              i.startswith("ylabel_") or i.startswith("charaR_") or i.startswith(
        #                  "charao_") or i.startswith("charas_")]
        # # 待学习内容列
        # charapd = respd[charalist]
        # learn_list = [i for i in charalist if i.startswith("charas_")]
        # learn_pd = pd.DataFrame()
        # for i1 in learn_list:
        #     learn_pd = pd.concat([learn_pd, charapd[(charapd[i1] == -1)]], axis=0, join='outer')
        # # 排序
        # learn_pd.sort_index(axis=0, ascending=True, inplace=True)
        # # 去重
        # learn_pd.drop_duplicates(inplace=True)
        # # 去空
        # learn_pd.dropna(inplace=True)
        # return learn_pd

    # 深度学习所需特征
    def deeplearn_charas(self, pdobj, parajson):
        # 只有基本输入，特征是学出来的。
        dclass = Sequence_chara(pdobj, None)
        mainlist = ["open", "high", "close", "low", "volume"]
        for stock in dclass.pf:
            dclass.pf[stock] = dclass.pf[stock][mainlist]
            for i2 in mainlist:
                dclass.pf[stock][i2] = dclass.pf[stock][i2].pct_change()
                dclass.pf[stock][i2] = np.log(dclass.pf[stock][i2])
        return dclass.pf

    # 机器学习所需特征
    def mla_charas(self, pdobj, parajson):
        # 全面的人工特征
        parajson = {
            "avenlist": [5, 20, 60],
            "labellist": [1, 2, 4, 8, 16],
        }
        dclass = Sequence_chara(pdobj, parajson)
        respd = dclass.pf
        respd = respd[respd.columns]
        del respd["turnover"]
        respd.dropna(inplace=True)
        return respd


def main():
    pass


if __name__ == '__main__':
    # 1. 测试
    main()
    # 随机森林输入
