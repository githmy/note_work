# --coding:utf-8
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

filepath = os.path.join("../dataset/Internet Advertisements/Data Folder", "ad.data")


# 对DataFrame的列做数据转换
def Converter_number(x):
    try:
        return np.float64(x)
    except ValueError:
        return np.nan


# 字典推导式
converters = {key: Converter_number for key in range(1558)}
converters[1558] = lambda x: 1 if x.strip() == 'ad.' else 0
ads = pd.read_csv(filepath, header=None, converters=converters)
print(ads[:5])


# 我们来看看前4个特征的分布
fig = plt.figure()
df = (ads[0].sort_values().values)[:, np.newaxis]
grid_param = {
    'bandwidth': list(range(1, 31))
}
kde_grid = GridSearchCV(KernelDensity(), grid_param)
kde = kde_grid.fit(df).best_estimator_
print(kde_grid.best_params_)
plt.subplot(221)
plt.plot(df[:, 0], np.exp(kde.score_samples(df)), '-')

df = (ads[1].sort_values().values)[:, np.newaxis]
grid_param = {
    'bandwidth': list(range(1, 31))
}
kde_grid = GridSearchCV(KernelDensity(), grid_param)
kde = kde_grid.fit(df).best_estimator_
print(kde_grid.best_params_)
plt.subplot(222)
plt.plot(df[:, 0], np.exp(kde.score_samples(df)), '-')

df = (ads[2].sort_values().values)[:, np.newaxis]
grid_param = {
    'bandwidth': list(range(1, 11))
}
kde_grid = GridSearchCV(KernelDensity(), grid_param)
kde = kde_grid.fit(df).best_estimator_
print(kde_grid.best_params_)
plt.subplot(223)
plt.plot(df[:, 0], np.exp(kde.score_samples(df)), '-')

df = (ads[3].sort_values().values)[:, np.newaxis]
grid_param = {
    'bandwidth': np.linspace(0.01, 0.1, 10)
}
kde_grid = GridSearchCV(KernelDensity(), grid_param)
kde = kde_grid.fit(df).best_estimator_
print(kde_grid.best_params_)
plt.subplot(224)
plt.plot(df[:, 0], np.exp(kde.score_samples(df)), '-')

# 可用来对类别变量进行计数
from collections import Counter

df1558 = ads[1558].values
# c = Counter(df1558)
# plt.bar(list(c.keys()),list(c.values()))
fig = plt.figure()
plt.hist(df1558, bins=2, rwidth=0.8)
plt.show()

if __name__ == "__main__":
    print("OK")