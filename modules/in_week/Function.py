"""
"""
import pandas as pd
import copy
import matplotlib.pyplot as plt
import os
import numpy as np

# 计算涨跌幅并区分到各个工作日
def cal_and_classify(source_data, time='candle_end_time', open='open', high='high', low='low', close='close'):
    temp = source_data.copy()
    temp['涨跌幅'] = temp[close] / temp[close].shift(1) - 1
    temp.dropna(subset=['涨跌幅'], inplace=True)
    temp['星期'] = temp[time].dt.dayofweek
    return temp


def classify_date(source_data):
    mon_df = source_data[source_data['星期'] == 0]
    tue_df = source_data[source_data['星期'] == 1]
    wed_df = source_data[source_data['星期'] == 2]
    thu_df = source_data[source_data['星期'] == 3]
    fri_df = source_data[source_data['星期'] == 4]
    return mon_df, tue_df, wed_df, thu_df, fri_df


# 计算定投
def regular_investment(source_data, close='close', time='candle_end_time', capital=1000, rate=0.00006, ):
    temp = source_data.copy()
    # ===计算累计投入资金
    temp['每次投入资金'] = capital  # 每个周期投入10000元买币
    temp['累计投入资金'] = temp['每次投入资金'].cumsum()  # 至今累计投入的资金，cumulative_sum

    # ===计算累计买入数量
    temp['每次买入数量'] = temp['每次投入资金'] / temp[close] * (1 - rate)  # 每个周期买买入的数量，扣除了手续费（此处手续费计算有近似）
    temp['累计买入数量'] = temp['每次买入数量'].cumsum()  # 累计买入的数量

    # ===计算买入的市值
    temp['平均持有成本'] = temp['累计投入资金'] / temp['累计买入数量']
    temp['持有市值'] = temp['累计买入数量'] * temp[close]
    temp.dropna(subset=['持有市值'], inplace=True)
    print(temp[['candle_end_time', '星期', 'close', '累计投入资金', '持有市值']].tail(1))

    # ===输出数据
    return temp[[time, close, '累计投入资金', '持有市值', '平均持有成本']]


# 均线区分上涨下跌
def classify_rise_fall_by_mean(source_data, mean_days=20):
    temp = source_data.copy()
    temp.reset_index(drop=True, inplace=True)
    temp.loc[(temp['close'] > temp['close'].rolling(mean_days, min_periods=1).mean()), '上涨市_mean'] = True
    temp['上涨市_mean'].fillna(value=False, inplace=True)
    return temp
