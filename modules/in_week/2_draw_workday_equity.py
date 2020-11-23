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

# 绘制各个工作日涨跌幅
temp = df[['candle_end_time', '星期', 'close']].copy()

mon_df = df[df['星期'] == 0]
tue_df = df[df['星期'] == 1]
wed_df = df[df['星期'] == 2]
thu_df = df[df['星期'] == 3]
fri_df = df[df['星期'] == 4]

mon_df['equity_mon'] = (mon_df['涨跌幅'] + 1).cumprod()
tue_df['equity_tue'] = (tue_df['涨跌幅'] + 1).cumprod()
wed_df['equity_wed'] = (wed_df['涨跌幅'] + 1).cumprod()
thu_df['equity_thu'] = (thu_df['涨跌幅'] + 1).cumprod()
fri_df['equity_fri'] = (fri_df['涨跌幅'] + 1).cumprod()

# 合并
temp['equity_mon'] = mon_df['equity_mon']
temp['equity_tue'] = tue_df['equity_tue']
temp['equity_wed'] = wed_df['equity_wed']
temp['equity_thu'] = thu_df['equity_thu']
temp['equity_fri'] = fri_df['equity_fri']

# 填充
equity_list = ['equity_mon', 'equity_tue', 'equity_wed', 'equity_thu', 'equity_fri']

for i in equity_list:
    temp[i].fillna(method='ffill', inplace=True)
    temp[i].fillna(value=1, inplace=True)
    plt.plot(temp['candle_end_time'], temp[i], label=i)
plt.legend(loc='best')
plt.show()

