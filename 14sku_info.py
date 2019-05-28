#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import itertools
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random

infilename1 = "../data/tt_orderdetail7.csv"
detail_df = pd.read_csv(infilename1, header=0, encoding="utf8", sep=',')
infilename2 = "../data/tt_orderattribute8.csv"
attri_df = pd.read_csv(infilename2, header=0, encoding="utf8", sep=',')

detail_df = detail_df[["OrderID", "CommodityID"]]
attri_df = attri_df[["SubmitDate", "OrderID"]]

oriid = list(set(detail_df["CommodityID"]))
newid = list(range(len(oriid)))
orinew_json = {}
newori_json = {}
for i1 in zip(oriid, newid):
    orinew_json[i1[0]] = i1[1]
    newori_json[i1[1]] = i1[0]


# new_pd =pd.merge(attri_df, detail_df, on="OrderID", how="left")
# new_pd.set_index("SubmitDate", inplace=True)
# new_pd.sort_index(axis=0, ascending=True, inplace=True)
# new_pd = new_pd.iloc[0:1000, :]


def sort_df2(data):
    lists = list(data["CommodityID"])
    return lists


info_new = pd.DataFrame()
# print(new_pd)
info_new["skus"] = detail_df.groupby(["OrderID"]).apply(sort_df2)
info_new["length"] = info_new["skus"].map(len)
info_new = pd.merge(attri_df, info_new, on="OrderID", how="left")
info_new.set_index("SubmitDate", inplace=True)

# 数据窗口分析部分 起
datefpoint = "2017-11-25"
datetpoint = "2019-11-30"
tt_info_new = info_new[datefpoint:datetpoint]
tt_info_new.index = pd.to_datetime(tt_info_new.index)

min_list = []
upd_list = []
new_sku_t = []
min_list2d = []
upd_list2d = []
for c99 in range(10):
    min_list.append(int(2 ** c99))
for c99 in range(6):
    upd_list.append(int(2 ** c99))
min_list.pop(0)
upd_list.pop(0)
print("min_list:", min_list)
print("upd_list:", upd_list)
for i99 in min_list:
    checknew_unitmin = i99
    for i98 in upd_list:
        startt = time.time()
        checknew_updatday = i98
        print("unit min %s update %s." % (checknew_unitmin, checknew_updatday))


        def find_new(data):
            data = list(set(itertools.chain(*[i1 for i1 in list(data.values)])))
            return data


        resam_info_new = tt_info_new["skus"].resample(str(checknew_unitmin) + 'T').apply(find_new)
        sku_addn = np.zeros((resam_info_new.shape[0]))
        find_n_seg = int(24 * 60 * checknew_updatday / checknew_unitmin)
        for i1, c1 in enumerate(resam_info_new):
            if i1 > find_n_seg:
                bdata = list(set(itertools.chain(*[i2 for i2 in list(resam_info_new.iloc[i1 - find_n_seg:i1])])))
                sku_addn[i1] = len([i2 for i2 in list(resam_info_new.iloc[i1]) if i2 not in bdata])
            else:
                sku_addn[i1] = None
        showpd = pd.Series(sku_addn)
        showpd.index = resam_info_new.index

        counterlist = []
        counterb = 0
        for i1, c1 in enumerate(showpd):
            if str(c1) != "nan" and int(c1) != 0:
                counterlist.append(i1 - counterb)
                counterb = i1
        counterlist.pop(0)
        counterlist_length = len(counterlist)
        final_inte = sum(counterlist) / counterlist_length * checknew_unitmin / 60
        new_sku_t.append(final_inte)
        min_list2d.append(checknew_unitmin)
        upd_list2d.append(checknew_updatday)
        print("use time %s mins." % ((time.time() - startt) / 60))
        print("average time for new sku is %s hours." % final_inte)

# fin_showpd = pd.Series(new_sku_t)
# fin_showpd = pd.Series(checknew_unitmin)
# fin_showpd = pd.Series(checknew_updatday)

# fin_showpd.index = pd.Series(min_list)
# plt.rcParams['figure.figsize'] = (16.0, 8.0)
# fin_showpd.plot()
# print(new_sku_t)
min_list2, upd_list2 = np.meshgrid(upd_list, min_list)
new_sku_t2 = np.array(new_sku_t).reshape((9, 5))
# print(min_list2)
# print(upd_list2)
# print(new_sku_t2)

# %matplotlib
# get_ipython().run_line_magic('matplotlib', '')

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(min_list2, upd_list2, new_sku_t2, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()

# 数据窗口分析部分 止

# 数据统计货架分布部分 起
datefpoint = "2018-11-25"
datetpoint = "2018-11-30"
ttc_info_new = info_new[datefpoint:datetpoint]

skulen = len(orinew_json)


def hot_deal(data):
    hotnp = np.zeros(skulen)
    for i1 in data:
        hotnp[orinew_json[i1]] = 1
    return hotnp


ttc_info_new["one_hot"] = ttc_info_new["skus"].map(hot_deal)
print(ttc_info_new)

dataobj = np.array([list(i1) for i1 in list(ttc_info_new["one_hot"].values)])

# print(dataobj.shape)
# dataobj = dataobj[0:dataobj.shape[0]//10,:]
# print(dataobj.shape)
# startt = time.time()
shelf_total_n = 40
clss = KMeans(n_clusters=shelf_total_n, init='k-means++')
y_hat = clss.fit_predict(dataobj)
# print(type(y_hat))
# print(len(y_hat))
# print(y_hat)
# print(time.time()-startt)


ttc_info_new["shelf"] = y_hat


def sort_func(data):
    lists = list(data["skus"])
    lists = list(itertools.chain(*[i1 for i1 in data["skus"]]))
    return lists


final_pd = pd.DataFrame()
# print(new_pd)
# 同一货架的 不同订单 出现商品的次数，没有累计同一订单的商品数
final_pd["skulist"] = ttc_info_new.groupby(["shelf"]).apply(sort_func)
# 同一货架的 商品的种类
final_pd["skuset"] = final_pd["skulist"].map(set)
final_pd["skulistlength"] = final_pd["skulist"].map(len)
final_pd["skusetlength"] = final_pd["skuset"].map(len)
alist = []
for i1 in final_pd["skulist"]:
    alist.extend(i1)
print(final_pd)


# 统计商品数量
def jsform(data):
    skujson = {}
    for i1 in data:
        if i1 not in skujson:
            skujson[i1] = 1
        else:
            skujson[i1] += 1
    return skujson


final_pd["skujson"] = final_pd["skulist"].map(jsform)


def jstop(data):
    data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    llst = [i1[0] for i1 in data]
    return llst


final_pd["skujson_topn"] = final_pd["skujson"].map(jstop)

print(final_pd["skujson_topn"])

# skujson_top n 重复的比重

all_sku_lists = list(set(alist))
all_sku_lenth = len(set(alist))
same_list = []
heatnp_same_self_ratio = np.zeros((final_pd.shape[0], final_pd.shape[0]))
heatnp_same_all_ratio = np.zeros((final_pd.shape[0], final_pd.shape[0]))
heatnp_sku_all_ratio = np.zeros((final_pd.shape[0], final_pd.shape[0]))
topn = 30
# 按规律从大到小
# ttopnp = [list(i1[0:topn]) for i1 in final_pd["skujson_topn"].values]
# 按规律 随机
ttopnp = [random.sample(i1, topn) if len(i1) >= topn else i1 for i1 in final_pd["skujson_topn"].values]
# # 按订单出现次数随机
# ttopnp = [random.sample(alist, topn)  for i1 in range(shelf_total_n)]
main_sku_list = list(set(itertools.chain(*[i1 for i1 in ttopnp])))
# print(type(ttopnp))
# print(ttopnp)
# for i1,l1 in enumerate(ttopnp):
#     heatnp_sku_all_ratio[i1]= len(l1)/all_sku_lenth
for i1, l1 in enumerate(ttopnp):
    for i2, l2 in enumerate(ttopnp):
        heatnp_same_self_ratio[i1, i2] = len([i for i in l1 if i in l2]) / len(l1)
        heatnp_same_all_ratio[i1, i2] = len([i for i in l1 if i in l2]) / all_sku_lenth
        heatnp_sku_all_ratio[i1, i2] = len(l1) / all_sku_lenth
print("main_sku_lenth/all_sku_lenth:", len(main_sku_list) / all_sku_lenth)
print("all_sku_lenth:", all_sku_lenth)

plt.rcParams['figure.figsize'] = (16.0, 8.0)  # 设置figure_size尺寸
plt.figure(1)
sns.heatmap(heatnp_same_self_ratio, annot=False)
plt.draw()
plt.figure(2)
sns.heatmap(heatnp_same_all_ratio, annot=False)
plt.draw()
plt.figure(3)
sns.heatmap(heatnp_sku_all_ratio, annot=False)
plt.draw()
plt.show()

# 数据统计货架分布部分 止