"""
"""
import pandas as pd
from Function import *

pd.set_option('expand_frame_repr', False)
# 读取历史的沪深300数据
df = pd.read_csv('sh000300.csv', encoding='gbk', parse_dates=['candle_end_time'])
df = df[df['candle_end_time'] >= pd.to_datetime('20060101')]
df = cal_and_classify(df)

# 计算并绘制定投曲线

# 1、将df按照工作日进行划分，并将df复制一个副本出来
mon_df, tue_df, wed_df, thu_df, fri_df = classify_date(df)
temp = df[['candle_end_time', '星期', 'close']].copy()

# 2、计算每个工作日的定投

mon_df = regular_investment(mon_df)
tue_df = regular_investment(tue_df)
wed_df = regular_investment(wed_df)
thu_df = regular_investment(thu_df)
fri_df = regular_investment(fri_df)

# 3、将每个工作日的持有市值合并到副本中
temp['investment'] = mon_df['累计投入资金']
temp['equity_mon'] = mon_df['持有市值']
temp['equity_tue'] = tue_df['持有市值']
temp['equity_wed'] = wed_df['持有市值']
temp['equity_thu'] = thu_df['持有市值']
temp['equity_fri'] = fri_df['持有市值']

# 填充
equity_list = ['investment', 'equity_mon', 'equity_tue', 'equity_wed', 'equity_thu', 'equity_fri']

for i in equity_list:
    temp[i].fillna(method='ffill', inplace=True)
    temp[i].fillna(value=1, inplace=True)
    plt.plot(temp['candle_end_time'], temp[i], label=i)
plt.legend(loc='best')
plt.show()
