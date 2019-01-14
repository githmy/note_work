# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import os
import numpy as np

df = pd.DataFrame()
# 列聚类统计
df['label_L4'].value_counts()

# 过去12的平均值
moving_avg = pd.rolling_mean(ts_log, 12)
TP.rolling(window=ndays).mean()
# 指数加权移动平均法
expwighted_avg = pd.ewma(ts_log, halflife=12)
# 一阶差分
ts_log_diff = ts_log - ts_log.shift()

# 建立空内容
# orderl_pd = pd.DataFrame(data={})
# orderl_pd = pd.DataFrame({"phone":[111,222],"age":[3,5]},index=["first","second"])
# orderl_pd = pd.DataFrame(np.random(2,2),index=["first","second"],columns=["phone","age"])
# index=[10, 20, 30, 40, 50]

# orderl_pd = pd.date_range("2016 Jul 15 10:55", periods=10,freq='M')
# orderl_pd = pd.date_range("2016-01-02 10:50:00", periods=10,freq='2h12min')
# orderl_pd = pd.period_range("2016 Jul 15 10:55", periods=10,freq='60T')

# orderl_pd = pd.Timestamp("2016-01-02 10:50:00", tzinfo="shanghai")

# 时间过滤采样
ts = pd.Series(list(range(50)), index=pd.date_range("2016 Jul 15 10:55", periods=10, freq='60T'))
ts.asfreq("45Min", method="ffill")

# 时间段 取均值
df1 = pd.DataFrame()
DF = df.set_index(df1['time_slot1'])
DF.index = pd.to_datetime(DF.index, unit='ns')
ticket = DF.ix[:, ['all_time']]
# 以20分钟为一个时间间隔，求出所有间隔的平均时间
A_2analysisResult = ticket.all_time.resample('20min').mean()

# 索引设为列
# orderl_pd.reset_index(level=0, inplace=True)  # （the first）index 改为 column
# orderl_pd.reset_index()  # 丢弃原有的重赋值

# 列设为索引
# orderl_pd.set_index(i, inplace=True)

# 列改名
# plotlist = [data_list[i2].rename(columns={"close": i2})[i2] for i2 in orderl_pd[numf:numt]["index"]]

# 列合并
# plotlist_pd = pd.concat(plotlist, axis=1)

# # axis=0 是行拼接，拼接之后行数增加，列数也根据join来定，join='outer'时，列数是两表并集。同理join='inner',列数是两表交集。
# concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None,
#        verigy_integrity=False)
# print(df)

# join
# train_df = pd.merge(train_df, gdf, on="card_id", how="left")

# 列行索引
# orderl_pd[numf:numt]["index"]
# plotlist_pd.loc[i, i2]
# df.loc[[index],[colunm]] 通过位置选择数据
# df.loc[0:3, ['a', 'b']]

# 多行索引
# pddata.loc[indexres,]

# 选行按序号
# df.iloc[0, :]

# 行列索引
# df.iat[2, 3]
# df.at["gamma", "d"]


# # 遍历每一行
# for indexs in data.iterrows():
#     row[0], row[1]
# for indexs in data.index:
#     data.loc[indexs].values[0:-1]
# for indexs in data.rows:
#     row['c1'], row['c2']
# for row in df.itertuples(index=True, name='Pandas'):
#     print getattr(row, "c1"), getattr(row, "c2")

# 空值丢弃
df.dropna(subset=['closeprice'], inplace=True)
# 空值丢弃阈值
df.dropna(thresh=6, inplace=True)
# 空值填充
df.fillna(value=20181010, inplace=True)
df.fillna(value=20181010)
# 先向下填充
df.fillna(method='ffill', inplace=True)
# 再向上填充
df.fillna(method='bfill', inplace=True)

# 无穷处理
df.replace([np.inf, -np.inf], np.nan)

# one hot 转化
pd.get_dummies(df, columns=['category_2', 'category_3'])

# 列行数据类型 空值处理
# orderl_pd[[i]] = orderl_pd[[i]].fillna(1e6).astype(int)
# typess={'a': np.float64, 'b': np.int32}
# df2 = pd.read_csv(self.file_liquids_mount, header=0, encoding="utf8", sep='\t', dtype=typess)

# 判断在之内
df[df['secid'].isin([38, 24, 33])]

# 所有列名
# plotlist_pd.columns

# 列间求和
# plotlist_pd['ave' + str(lenthcolumn)] = plotlist_pd.apply(lambda x: x.sum() / lenthcolumn, axis=1)
# plotlist_pd['log(ave)' + str(lenthcolumn)] = np.log(plotlist_pd['ave' + str(lenthcolumn)])
# # 行间求和
# plotlist_pd.loc['Row_sum'] = plotlist_pd.apply(lambda x: x.sum() / len(liquids_pd.columns))
# print(plotlist_pd)

# 索引排序
# orderl_pd.sort_index(axis=0, ascending=True, inplace=True)

# 列值排序
# df.sort_values("age", ascending=False)

# 排序
df.sort(columns=["age", "tradedate"], ascending=[True, False])

# 分组 (key1+key2都不同)
# means = df['data1'].groupby([df['key1'], df['key2']]).mean()
# group 批量函数
# means = df['data1'].groupby([df['key1'], df['key2']]).mean()
agg_func = {
    'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
    'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
}
grouped = df.groupby(['card_id', 'month_lag'])
intermediate_group = grouped.agg(agg_func)
final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])


# 自定义group操作
def sort_df2(data):
    data = data.sort_values(by='df2', ascending=False)  # df2：品种列 ascending：排序方式
    return data


# groupby以及apply的结合使用
group = df.groupby(df['df1']).apply(sort_df2)

# 列值排序的序号
# orderl_pd[i] = liquids_pd[i].rank(ascending=1, method='first')

# # 添加增列
# datalists[i1].insert(1, "liquid", 1.0)
# datalists[i1].insert(0, "liquid", 1.3)
# # 添加增行
# row = pd.DataFrame([[sstrr[6], 0, sstrr[7]]], columns=["评论内容", "评论数", "点赞数"])
# pddata.append(row, ignore_index=True)

# 写文件
# orderl_pd.to_csv(tmpo_path, index=False, header=True, encoding="utf-8")
# data.to_excel('sales_result.xls', sheet_name='Sheet1', index=False)
# writer = pd.ExcelWriter('a.xlsx')
# df1.to_excel(writer, sheet_name='sheet1')
# df2.to_excel(writer, sheet_name='sheet2')
# writer.save()
# 读文件
# df1 = pd.read_csv(self.file_liquids_order, header=None, encoding="utf8", dtype=str,sep='\t')
# data = pd.read_excel(io='Current.xls', sheet_name='Sheet1', header=0)

# 删除列
# del df2["date"]

# 改内容
# train.loc[i1, "comment_text"]=" ".join(jieba.cut(train.loc[i1, "comment_text"]))

# 筛选过滤
# single_pd.ix[daays-1:, :]
# df = pd.DataFrame({'BoolCol': [1, 2, 3, 3, 4], 'attr': [22, 33, 22, 44, 66]},
#                   index=[10, 20, 30, 40, 50])
# # print(df)
# a = df[(df.BoolCol == 3) & (df.attr == 22)]
# print(a)
# a = df[(df.BoolCol == 3) & (df.attr == 22)].index
# print(a)
# a = df[(df.BoolCol == 3) & (df.attr == 22)].index.tolist()
# print(a)
# pddata = pddata[(~pddata["评论内容"].isnull())]
# pddata = pddata[(~pddata["评论内容"].notnull())]

# print(df['2013'].head(2)) # 获取2013年的数据
# print(df['2013'].tail(2)) # 获取2013年的数据
# 
# print(df['2016':'2017'].head(2))  #获取2016至2017年的数据
# print(df['2016':'2017'].tail(2))  #获取2016至2017年的数据

# 筛选赋值
class_data.loc[class_data['分类'] == 1, 'postive'] = 1
class_data.loc[class_data['分类'] == 0, 'neutral'] = 1
# 根据条件赋值
df['panduan'] = df.city.apply(lambda x: 1 if 'ing' in x else 0)

# print(df['2013'].head(2)) # 获取2013年的数据
# print(df['2013'].tail(2)) # 获取2013年的数据
#
# print(df['2016':'2017'].head(2))  #获取2016至2017年的数据
# print(df['2016':'2017'].tail(2))  #获取2016至2017年的数据

# 去重
# 按secid去重，保留最后的
pd.drop_duplicates(subset='secid', keep='last', inplace=True)

# 1. 序列处理，平移
# ts_lag = ts.shift()
# # 2. 均线
# ts_lag = ts.rolling(window=20)

# 每日涨幅
# df['close'].pct_change()


# tsv 读写
# infilename = "C:\\Users\\john\\Desktop\\train.tsv"
# # df2 = pd.read_csv(infilename, header=None, encoding="utf8", sep='\t')
# # print(df2.head()
# quotechar = None
# with tf.gfile.Open(infilename, "r") as f:
#     reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#     lines = []
#     for line in reader:
#         if len(line) == 6:
#             lines.append(line)
#             # print(lines)
#
# with open('eggs.tsv', 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     # writer.writerows(someiterable)
#     for i in lines:
#         # writer.writerow(
#         #     [i[0].encode("utf-8").decode("gbk"), i[3].encode("utf-8").decode("gbk"), i[4].encode("utf-8").decode("gbk"),
#         #      i[5].encode("utf-8").decode("gbk")])  # 写入csv文件的表头
#         # writer.writerow([i[0].encode("utf-8"), i[3].encode("utf-8"), i[4].encode("utf-8"), i[5].encode("utf-8")])  # 写入csv文件的表头
#         writer.writerow([str(i[0].encode("utf-8")), i[3].encode("utf-8"), i[4].encode("utf-8"), i[5].encode("utf-8")])  # 写入csv文件的表头
#         # writer.writerow([i[0], i[3], i[4], i[5]])  # 写入csv文件的表头

# 列为行的聚类索引
#     three  two  one
# AA      0    1    2
# BB      3    4    5
df.unstack()
df.unstack(0)
df.unstack(-1)
# three  AA    0
#        BB    3
# two    AA    1
#        BB    4
# one    AA    2
#        BB    5
# 转置 分两步
# 1.
df2 = df.stack()
df = df2.unstack(0)

# 区间切割
# bins 为整数左右延长1% bins刀按输入的极限等比例，数组，按位置切。 右闭区间
pandas.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False)
# qcut 按整数位切割
pandas.qcut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False)

# 批量操作
df["aaa"].map(func)
df[["aaa"]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# 日期转化
# 把时间列标准化时间格式
df['time_slot1'] = pd.to_datetime(df['time_slot1'])
# 输出这一天是周中的第几天，Monday=0, Sunday=6
df['dayofweek'] = df['time_slot1'].dt.dayofweek
df['daynameofweek'] = df['time_slot1'].dt.weekday_name

# 内存监控
start_mem = df.memory_usage().sum() / 1024 ** 2
end_mem = df.memory_usage().sum() / 1024 ** 2

# 显示设置
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.set_option('display.max_columns', 500)


def nomal_use():
    # 描述属性的各列统计值
    df.describe()
    df.describe()
    # 统计某列的数量分布
    df["age"].hist()
    # 某列的唯一值
    df["embarked"].unique()
