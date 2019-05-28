import pandas as pd
import numpy as np

# ~)01. 大数据量处理

# ~)02. normalize
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(np.array(X_train1))


# ~)03. 过采样
def over_sample(pdin, col):
    pdin[col].groupby([col]).count()
    return pdin


def over_sample2():
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    over_sampler = SMOTE(random_state=0)
    features = pd.DataFrame()["datas"]
    labels = pd.DataFrame()["labels"]
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=0)
    os_feature, os_label = over_sampler.fit_sample(feature_train, label_train)
    return os_feature, os_label


# ~)04. 数据分割
# 时间序列
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)
TimeSeriesSplit(max_train_size=None, n_splits=3)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# TRAIN: [0] TEST: [1]
# TRAIN: [0 1] TEST: [2]
# TRAIN: [0 1 2] TEST: [3]

# ~)05. 数据填充
titanic_test = pd.read_csv("test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

# ~)06. 数据过滤
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# ~)07. 数据合并
