import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# 线性回归模型 预测结果
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())

# 线性回归
linear_model = LinearRegression()
linear_model.fit(admissions[["gpa"]], admissions["admit"])

# 逻辑回归
logistic_model = LogisticRegression()
logistic_model.fit(admissions[["gpa"]], admissions["admit"])
pred_probs = logistic_model.predict_proba(admissions[["gpa"]])
plt.scatter(admissions["gpa"], pred_probs[:, 1])
plt.show()

# 增量学习
# 在线学习
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss='log')
# 用数据集训练
clf.partial_fit(X, y,classes=None, sample_weight=None)
# 当我们有了新数据之后，可以在原基础上更新模型
clf.partial_fit(X_new, y_new)
# partial_fit的模型使用方法也是和正常模型一样的，直接用predict或者predict_proba
y_pred = clf.predict_proba(X_test)


# # Classification
# sklearn.naive_bayes.MultinomialNB
# sklearn.naive_bayes.BernoulliNB
# sklearn.linear_model.Perceptron
# sklearn.linear_model.SGDClassifier
# sklearn.linear_model.PassiveAggressiveClassifier
# sklearn.neural_network.MLPClassifier

# # Regression
# sklearn.linear_model.SGDRegressor
# sklearn.linear_model.PassiveAggressiveRegressor
# sklearn.neural_network.MLPRegressor

# # Clustering
# sklearn.cluster.MiniBatchKMeans
# sklearn.cluster.Birch
# Decomposition / feature Extraction
# sklearn.decomposition.MiniBatchDictionaryLearning
# sklearn.decomposition.IncrementalPCA
# sklearn.decomposition.LatentDirichletAllocation

# # Preprocessing
# sklearn.preprocessing.StandardScaler
# sklearn.preprocessing.MinMaxScaler
# sklearn.preprocessing.MaxAbsScaler

# # 其他框架参考
# https://dask.pydata.org/en/latest/
