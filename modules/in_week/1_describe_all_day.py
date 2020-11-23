"""
"""
import pandas as pd
from Function import *
pd.set_option('expand_frame_repr', False)
# 读取历史的沪深300数据

df = pd.read_csv('sh000300.csv', encoding='gbk', parse_dates=['candle_end_time'])
df['涨跌幅'] = df['close'] / df['close'].shift(1) - 1
df = df[df['candle_end_time'] >= pd.to_datetime('20060101')]

# 计算工作日
df['星期'] = df['candle_end_time'].dt.dayofweek
df['星期'] += 1

# 统计各个工作日的均值，涨跌幅等特征
result = df.groupby('星期')['涨跌幅'].describe()
temp1 = df.groupby('星期')['涨跌幅'].size()
temp2 = df[df['涨跌幅'] > 0].groupby('星期')['涨跌幅'].size()
result['胜率'] = temp2 / temp1
print(result.T)
