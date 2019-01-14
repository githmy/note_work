# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tushare as ts

if __name__ == '__main__':
    # 1. 数据选择
    # 1.1. 筛选集合
    stock_lis = ['300113', '300343', '300295', '300315']
    end = datetime.today()  # 开始时间结束时间，选取最近一年的数据
    start = datetime(end.year - 1, end.month, end.day)
    end = str(end)[0:10]
    start = str(start)[0:10]
    df = pd.DataFrame()
    for stock in stock_lis:
        closing_df = ts.get_hist_data(stock, start, end)['close']
        df = df.join(pd.DataFrame({stock: closing_df}), how='outer')
    tech_rets = df.pct_change()
    print(df.head(3))
    print(df.tail(3))
    print(tech_rets.head(3))
    print(tech_rets.tail(3))

    # pearson相关热图
    rets = tech_rets.dropna()
    plt.figure(1)
    sns.heatmap(rets.corr(), annot=True)
    plt.draw()
    # plt.close(1)
    # 收益风险图
    plt.figure(2)
    plt.scatter(rets.mean(), rets.std())
    plt.xlabel('Excepted Return')
    plt.ylabel('Risk')
    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(label, xy=(x, y), xytext=(15, 15), textcoords='offset points',
                     arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3'))
    plt.draw()
    # plt.close(2)
    plt.show()
