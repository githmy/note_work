# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import logging
from sklearn.datasets import load_iris
from sklearn.datasets import make_gaussian_quantiles
import sklearn
import tushare as ts
import statsmodels.tsa.stattools as tsat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num, datestr2num
from datetime import datetime
# ! pip install statsmodels
import statsmodels as stats
import xgboost as xgb

cmd_path = os.getcwd()
data_path = os.path.join(cmd_path, "data")
data_path = os.path.join(data_path, "stock")
datalogfile = os.path.join(cmd_path, 'log')
datalogfile = os.path.join(datalogfile, 'data_analysis.log')

# 创建一个logger
logger1 = logging.getLogger('logger_out')
logger1.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(datalogfile)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger1.addFilter(filter)
logger1.addHandler(fh)
logger1.addHandler(ch)


def basic_Linear_Regression():
    """
    Y = aX + b这个公式里：
    Y - 因变量
    a - 斜率
    X - 自变量
    b - 截距
    """
    # Import Library
    # Import other necessary libraries like pandas, numpy...
    from sklearn import linear_model
    # Load Train and Test datasets
    # Identify feature and response variable(s) and values must be numeric and numpy arrays

    x_train = input_variables_values_training_datasets
    y_train = target_variables_values_training_datasets
    x_test = input_variables_values_test_datasets

    # Create linear regression object
    linear = linear_model.LinearRegression()

    # Train the model using the training sets and check score
    linear.fit(x_train, y_train)
    linear.score(x_train, y_train)

    # Equation coefficient and Intercept
    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)

    # Predict Output
    predicted = linear.predict(x_test)


def polynormal_Regression():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    pf = PolynomialFeatures(degree=2)
    x_2_fit = pf.fit_transform(x)  # 一元二次的方程就转化为二元一次的方程
    lrModel = LinearRegression()
    lrModel.fit(x_2_fit, y)
    lrModel.score(x_2_fit, y)  # 模型拟合程度
    x_2_predict = pf.fit_transform([[21], [22]])
    lrModel.predict([[21], [22]])


def basic_Logistic_Regression():
    """
    odds= p/ (1-p) = probability of event occurrence / probability of not event occurrence
    ln(odds) = ln(p/(1-p))
    logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
    """
    # Import Library
    from sklearn.linear_model import LogisticRegression
    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

    # Create logistic regression object

    model = LogisticRegression()

    # Train the model using the training sets and check score
    model.fit(X, y)
    model.score(X, y)

    # Equation coefficient and Intercept
    print('Coefficient: \n', model.coef_)
    print('Intercept: \n', model.intercept_)

    # Predict Output
    predicted = model.predict(x_test)


def basic_Decision_Tree():
    """
    
    """

    # Import Library
    # Import other necessary libraries like pandas, numpy...
    from sklearn import tree
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

    # 定义一个随机森林分类器
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y)
    scores.mean()
    # 定义一个极端森林分类器
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y)
    scores.mean()

    # # Create tree object
    # model = tree.DecisionTreeClassifier(
    #     criterion='gini')  # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini
    #
    # # model = tree.DecisionTreeRegressor() for regression
    #
    # # Train the model using the training sets and check score
    # model.fit(X, y)
    # model.score(X, y)

    # Predict Output
    # predicted = model.predict(x_test)


def basic_SVM():
    """

    """
    # Import Library
    from sklearn import svm
    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create SVM classification object

    model = svm.svc()  # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.

    # Train the model using the training sets and check score
    model.fit(X, y)
    model.score(X, y)

    # Predict Output
    predicted = model.predict(x_test)


def basic_Naive_Bayes():
    """

    """
    # Import Library
    from sklearn.naive_bayes import GaussianNB
    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

    # Create SVM classification object
    model = GaussianNB()  # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link

    # Train the model using the training sets and check score
    model.fit(X, y)

    # Predict Output
    predicted = model.predict(x_test)


def basic_KNN():
    """

    """
    # Import Library
    from sklearn.neighbors import KNeighborsClassifier

    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create KNeighbors classifier object model

    KNeighborsClassifier(n_neighbors=6)  # default value for n_neighbors is 5

    # Train the model using the training sets and check score
    model.fit(X, y)

    # Predict Output
    predicted = model.predict(x_test)


def basic_K_means():
    """

    """
    # Import Library
    from sklearn.cluster import KMeans

    # Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
    # Create KNeighbors classifier object model
    k_means = KMeans(n_clusters=3, random_state=0)

    # Train the model using the training sets and check score
    model.fit(X)

    # Predict Output
    predicted = model.predict(x_test)


def basic_Random_Forest():
    """
    如果训练集中有N种类别，则有重复地随机选取N个样本。这些样本将组成培养决策树的训练集。
    如果有M个特征变量，那么选取数m << M，从而在每个节点上随机选取m个特征变量来分割该节点。m在整个森林养成中保持不变。
    每个决策树都最大程度上进行分割，没有剪枝。
    """
    # Import Library
    from sklearn.ensemble import RandomForestClassifier
    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

    # Create Random Forest object
    model = RandomForestClassifier()

    # Train the model using the training sets and check score
    model.fit(X, y)

    # Predict Output
    predicted = model.predict(x_test)


def basic_Dimensionality_Reduction():
    """

    """
    # Import Library
    from sklearn import decomposition
    # Assumed you have training and test data set as train and test
    # Create PCA obeject
    pca = decomposition.PCA(n_components=k)  # default value of k =min(n_sample, n_features)
    # For Factor analysis
    # fa= decomposition.FactorAnalysis()
    # Reduced the dimension of training dataset using PCA

    train_reduced = pca.fit_transform(train)

    # Reduced the dimension of test dataset
    test_reduced = pca.transform(test)


def basic_Gradient_Boost():
    """

    """
    # Import Library
    from sklearn.ensemble import GradientBoostingClassifier
    # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Gradient Boosting Classifier object
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    # Train the model using the training sets and check score
    model.fit(X, y)
    # Predict Output
    predicted = model.predict(x_test)


def basic_Adaboost():
    """

    """
    pass


def basic_BayesianRegression():
    """

    """
    pass


def basic_HMM():
    pass


def iris_demo():
    iri = load_iris()
    print(iri)


# 推荐系统的库
def surprise_demo():
    ### 使用SVD
    from surprise import SVD, evaluate
    from surprise import Dataset, print_perf
    from surprise.model_selection import cross_validate

    # 默认载入movielens数据集
    music_data = Dataset.load_builtin('ml-100k')
    # k折交叉验证(k=3),此方法现已弃用
    music_data.split(n_folds=3)
    algo = SVD()
    perf = evaluate(algo, music_data, measures=['RMSE', 'MAE'])
    # 输出结果
    print_perf(perf)


def tmp_test():
    # 但只测试
    global data_path
    tpath = data_path
    coden = "000001"
    filePath = coden + "_" + "5" + ".csv"
    tmp_path = os.path.join(tpath, filePath)

    try:
        df = pd.read_csv(tmp_path, header=0, encoding="utf8")
        # print(df.head())
        aa = tsat.adfuller(df["open"], 1)
        print(aa)
        plt.plot(df["date"], df["open"], '--', lw=2)
        # plt.scatter(df["date"][0:20], df["open"][0:20], s=df["high"], c=["r","#0F0F0F0F"])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Mercator: %s' % coden)
        plt.grid(True)
        plt.show()
    except Exception as e:
        logger1.info("error with code: %s" % coden)
        logger1.info(e)


def basic_plot():
    def qq_polt():
        # 数据是否倾斜
        stats.mstats.skew(train["loss"]).data
        stats.mstats.skew(np.log(train["loss"])).data

    def box_cox():
        # 给数据点找到合适的函数，经过变换后，接近正太分布
        """
        y(l) = (yi^l-1)/l   如果 l!=0
        y(l) = ln(yi)       如果 l==0
        """
        from scipy.special import boxcox1p
        skewed_features = skewness.index
        alldata = []
        for feat in skewed_features:
            alldata[feat] = boxcox1p(alldata[feat], 0.15)

    def xgboost_demo():
        detrain = xgb.DMatrix(train_x, train["label"])
        detrain = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=3, seed=0, feval=xg_eval_mae, maximize=False,
                         early_stopping_rounds=10)

    def arima_demo():
        # 时间序列
        """
        ARIMA(p,d,q)阶数的确定。
                       ACF           PACF
        AR(p)      衰减震荡趋于零，p阶后截断
        MA(q)      q阶后截断,     衰减震荡趋于零
        ARMA(p,q)  q阶后截断,     p阶后截断
        
        选择标准 AIC BIC
        AIC 赤池信息准则     AIC=2k-2ln(L)
        BIC 贝叶斯信息准则   BIC=kln(n)-2ln(L)
        k 模型数量， n 样本数量， L 似然函数
        """
        import statsmodels.api as sm

        mod = sm.tsa.statespace.SARIMAX(y,
                                        order=(1, 1, 1),
                                        freq="W-MON",
                                        seasonal_order=(1, 1, 1, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()
        print(results.summary().tables[1])


if __name__ == '__main__':
    # http://scikit-learn.org/stable/auto_examples/
    # http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html
    iris_demo()
    # 1. 单策略学习
    # tmp_test()
    # 1.1 单策略特征提取
    # 1.2 特征学习筛选
    # 2. 策略集筛选评估
    # 3. 应用查结果。资产组合，CAPM model。sigma=100%
    # 4. 回测
    # 5.1 规律猜测，靶定验证(流通市值排列，n日后的变化:单点时间 和 采样时间, 线性回归)
    # 5.2 规律猜测，靶定验证(交叉点，mn差,配合导数值, 逻辑回归)
    # 5.3 规律猜测，靶定验证(均值，1.5倍波动方差回归才操作, 逻辑回归)
    # 5.4 规律猜测，靶定验证(流通市值排列，n日后的变化, 回归)
    # 5.4 规律猜测，靶定验证(流通市值 和 波动程度 之间的关系)
    # 5.4 规律猜测，散点判断趋势相似(线性回归，两两作对，最匹配的)
    # 5.5 时间序列，回归均值
    # 5.5 时间序列，动量
    # 5.5 时间序列，单只判断是否随机游走使用adfuller
