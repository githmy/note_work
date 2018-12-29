# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

'''
【说明】 
1.当前sklearn版本0.18 
2.sklearn自带的鸢尾花数据集样例： 
（1）样本特征矩阵（类型：numpy.ndarray） 
 [[ 6.7  3.   5.2  2.3] 
 [ 6.3  2.5  5.   1.9] 
 [ 6.5  3.   5.2  2. ] 
 [ 6.2  3.4  5.4  2.3] 
 [ 5.9  3.   5.1  1.8]] 
 每行是一个样本，矩阵行数=样本总数，矩阵列数=每个样本特征数 
 （2）样本类别矩阵（类型：numpy.ndarray） 
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
 2 2] 
 每个元素对应一个样本的类标 
 3.本地excel表的数据集样例： 
class0  p1  p2  p3  p4  p5  p6  p7 
0   0   0   0   1   0   0   0 
0   5   9   10  10  0   1   1 
0   0   1   1   0   0   1   0 
0   0   1   1   0   0   1   0 
每行是一个样本，每行第一个元素是样本所属类别，后续元素是样本的特征 
'''
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier


# 读取sklearn自带的数据集（鸢尾花）
def getData_1():
    iris = datasets.load_iris()
    X = iris.data  # 样本特征矩阵，150*4矩阵，每行一个样本，每个样本维度是4
    y = iris.target  # 样本类别矩阵，150维行向量，每个元素代表一个样本的类别


# 读取本地excel表格内的数据集（抽取每类60%样本组成训练集，剩余样本组成测试集）
# 返回一个元祖，其内有4个元素（类型均为numpy.ndarray）：
# （1）归一化后的训练集矩阵，每行为一个训练样本，矩阵行数=训练样本总数，矩阵列数=每个训练样本的特征数
# （2）每个训练样本的类标
# （3）归一化后的测试集矩阵，每行为一个测试样本，矩阵行数=测试样本总数，矩阵列数=每个测试样本的特征数
# （4）每个测试样本的类标
# 【注】归一化采用“最大最小值”方法。
def getData_2():
    fPath = 'F:/cleanData_dropSJS.csv'
    if os.path.exists(fPath):
        data = pd.read_csv(fPath, header=None, skiprows=1,
                           names=['class0', 'pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6'])
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data, data['class0'], test_size=0.4, random_state=0)
        min_max_scaler = preprocessing.MinMaxScaler()  # 归一化
        X_train_minmax = min_max_scaler.fit_transform(np.array(X_train1))
        X_test_minmax = min_max_scaler.fit_transform(np.array(X_test1))
        return (X_train_minmax, np.array(y_train1), X_test_minmax, np.array(y_test1))
    else:
        print('No such file or directory!')


# 读取本地excel表格内的数据集（每类随机生成K个训练集和测试集的组合）
# 【K的含义】假设一共有1000个样本，K取10，那么就将这1000个样本切分10份（一份100个），那么就产生了10个测试集
# 对于每一份的测试集，剩余900个样本即作为训练集
# 结果返回一个字典：键为集合编号（1train, 1trainclass, 1test, 1testclass, 2train, 2trainclass, 2test, 2testclass...），值为数据
# 其中1train和1test为随机生成的第一组训练集和测试集（1trainclass和1testclass为训练样本类别和测试样本类别），其他以此类推
def getData_3():
    fPath = 'F:/cleanData_dropSJS.csv'
    if os.path.exists(fPath):
        # 读取csv文件内的数据，
        dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1,
                                          names=['class0', 'pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5',
                                                 'pixel6']))
        # 获取每个样本的特征以及类标
        rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
        sampleData = []
        sampleClass = []
        for i in range(0, rowNum):
            tempList = list(dataMatrix[i, :])
            sampleClass.append(tempList[0])
            sampleData.append(tempList[1:])
        sampleM = np.array(sampleData)  # 二维矩阵，一行是一个样本，行数=样本总数，列数=样本特征数
        classM = np.array(sampleClass)  # 一维列向量，每个元素对应每个样本所属类别
        # 调用StratifiedKFold方法生成训练集和测试集
        skf = StratifiedKFold(n_splits=10)
        setDict = {}  # 创建字典，用于存储生成的训练集和测试集
        count = 1
        for trainI, testI in skf.split(sampleM, classM):
            trainSTemp = []  # 用于存储当前循环抽取出的训练样本数据
            trainCTemp = []  # 用于存储当前循环抽取出的训练样本类标
            testSTemp = []  # 用于存储当前循环抽取出的测试样本数据
            testCTemp = []  # 用于存储当前循环抽取出的测试样本类标
            # 生成训练集
            trainIndex = list(trainI)
            for t1 in range(0, len(trainIndex)):
                trainNum = trainIndex[t1]
                trainSTemp.append(list(sampleM[trainNum, :]))
                trainCTemp.append(list(classM)[trainNum])
            setDict[str(count) + 'train'] = np.array(trainSTemp)
            setDict[str(count) + 'trainclass'] = np.array(trainCTemp)
            # 生成测试集
            testIndex = list(testI)
            for t2 in range(0, len(testIndex)):
                testNum = testIndex[t2]
                testSTemp.append(list(sampleM[testNum, :]))
                testCTemp.append(list(classM)[testNum])
            setDict[str(count) + 'test'] = np.array(testSTemp)
            setDict[str(count) + 'testclass'] = np.array(testCTemp)
            count += 1
        return setDict
    else:
        print('No such file or directory!')


# K近邻（K Nearest Neighbor）
def KNN():
    clf = neighbors.KNeighborsClassifier()
    return clf


# 线性鉴别分析（Linear Discriminant Analysis）
def LDA():
    clf = LinearDiscriminantAnalysis()
    return clf


# 支持向量机（Support Vector Machine）
def SVM():
    clf = svm.SVC()
    return clf


# 逻辑回归（Logistic Regression）
def LR():
    clf = LogisticRegression()
    return clf


# 随机森林决策树（Random Forest）
def RF():
    clf = RandomForestClassifier()
    return clf


# 多项式朴素贝叶斯分类器
def native_bayes_classifier():
    clf = MultinomialNB(alpha=0.01)
    return clf


# 决策树
def decision_tree_classifier():
    clf = tree.DecisionTreeClassifier()
    return clf


# GBDT
def gradient_boosting_classifier():
    clf = GradientBoostingClassifier(n_estimators=200)
    return clf


# 计算识别率
def getRecognitionRate(testPre, testClass):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    return float(rightNum) / float(testNum)


# report函数，将调参的详细结果存储到本地F盘（路径可自行修改，其中n_top是指定输出前多少个最优参数组合以及该组合的模型得分）
def report(results, n_top=5488):
    f = open('F:/grid_search_rf.txt', 'w')
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            f.write("Model with rank: {0}".format(i) + '\n')
            f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]) + '\n')
            f.write("Parameters: {0}".format(results['params'][candidate]) + '\n')
            f.write("\n")
    f.close()


# 自动调参（以随机森林为例）
def selectRFParam():
    clf_RF = RF()
    param_grid = {"max_depth": [3, 15],
                  "min_samples_split": [3, 5, 10],
                  "min_samples_leaf": [3, 5, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": range(10, 50, 10)}
    # "class_weight": [{0:1,1:13.24503311,2:1.315789474,3:12.42236025,4:8.163265306,5:31.25,6:4.77326969,7:19.41747573}],
    # "max_features": range(3,10),
    # "warm_start": [True, False],
    # "oob_score": [True, False],
    # "verbose": [True, False]}
    grid_search = GridSearchCV(clf_RF, param_grid=param_grid, n_jobs=4)
    start = time()
    T = getData_2()  # 获取数据集
    grid_search.fit(T[0], T[1])  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)


# “主”函数1（KFold方法生成K个训练集和测试集，即数据集采用getData_3()函数获取，计算这K个组合的平均识别率）
def totalAlgorithm_1():
    # 获取各个分类器
    clf_KNN = KNN()
    clf_LDA = LDA()
    clf_SVM = SVM()
    clf_LR = LR()
    clf_RF = RF()
    clf_NBC = native_bayes_classifier()
    clf_DTC = decision_tree_classifier()
    clf_GBDT = gradient_boosting_classifier()
    # 获取训练集和测试集
    setDict = getData_3()
    setNums = len(setDict.keys()) / 4  # 一共生成了setNums个训练集和setNums个测试集，它们之间是一一对应关系
    # 定义变量，用于将每个分类器的所有识别率累加
    KNN_rate = 0.0
    LDA_rate = 0.0
    SVM_rate = 0.0
    LR_rate = 0.0
    RF_rate = 0.0
    NBC_rate = 0.0
    DTC_rate = 0.0
    GBDT_rate = 0.0
    for i in range(1, setNums + 1):
        trainMatrix = setDict[str(i) + 'train']
        trainClass = setDict[str(i) + 'trainclass']
        testMatrix = setDict[str(i) + 'test']
        testClass = setDict[str(i) + 'testclass']
        # 输入训练样本
        clf_KNN.fit(trainMatrix, trainClass)
        clf_LDA.fit(trainMatrix, trainClass)
        clf_SVM.fit(trainMatrix, trainClass)
        clf_LR.fit(trainMatrix, trainClass)
        clf_RF.fit(trainMatrix, trainClass)
        clf_NBC.fit(trainMatrix, trainClass)
        clf_DTC.fit(trainMatrix, trainClass)
        clf_GBDT.fit(trainMatrix, trainClass)
        # 计算识别率
        KNN_rate += getRecognitionRate(clf_KNN.predict(testMatrix), testClass)
        LDA_rate += getRecognitionRate(clf_LDA.predict(testMatrix), testClass)
        SVM_rate += getRecognitionRate(clf_SVM.predict(testMatrix), testClass)
        LR_rate += getRecognitionRate(clf_LR.predict(testMatrix), testClass)
        RF_rate += getRecognitionRate(clf_RF.predict(testMatrix), testClass)
        NBC_rate += getRecognitionRate(clf_NBC.predict(testMatrix), testClass)
        DTC_rate += getRecognitionRate(clf_DTC.predict(testMatrix), testClass)
        GBDT_rate += getRecognitionRate(clf_GBDT.predict(testMatrix), testClass)
    # 输出各个分类器的平均识别率（K个训练集测试集，计算平均）
    print
    print
    print
    print('K Nearest Neighbor mean recognition rate: ', KNN_rate / float(setNums))
    print('Linear Discriminant Analysis mean recognition rate: ', LDA_rate / float(setNums))
    print('Support Vector Machine mean recognition rate: ', SVM_rate / float(setNums))
    print('Logistic Regression mean recognition rate: ', LR_rate / float(setNums))
    print('Random Forest mean recognition rate: ', RF_rate / float(setNums))
    print('Native Bayes Classifier mean recognition rate: ', NBC_rate / float(setNums))
    print('Decision Tree Classifier mean recognition rate: ', DTC_rate / float(setNums))
    print('Gradient Boosting Decision Tree mean recognition rate: ', GBDT_rate / float(setNums))


# “主”函数2（每类前x%作为训练集，剩余作为测试集，即数据集用getData_2()方法获取，计算识别率）
def totalAlgorithm_2():
    # 获取各个分类器
    clf_KNN = KNN()
    clf_LDA = LDA()
    clf_SVM = SVM()
    clf_LR = LR()
    clf_RF = RF()
    clf_NBC = native_bayes_classifier()
    clf_DTC = decision_tree_classifier()
    clf_GBDT = gradient_boosting_classifier()
    # 获取训练集和测试集
    T = getData_2()
    trainMatrix, trainClass, testMatrix, testClass = T[0], T[1], T[2], T[3]
    # 输入训练样本
    clf_KNN.fit(trainMatrix, trainClass)
    clf_LDA.fit(trainMatrix, trainClass)
    clf_SVM.fit(trainMatrix, trainClass)
    clf_LR.fit(trainMatrix, trainClass)
    clf_RF.fit(trainMatrix, trainClass)
    clf_NBC.fit(trainMatrix, trainClass)
    clf_DTC.fit(trainMatrix, trainClass)
    clf_GBDT.fit(trainMatrix, trainClass)
    # 输出各个分类器的识别率
    print('K Nearest Neighbor recognition rate: ', getRecognitionRate(clf_KNN.predict(testMatrix), testClass))
    print('Linear Discriminant Analysis recognition rate: ', getRecognitionRate(clf_LDA.predict(testMatrix), testClass))
    print('Support Vector Machine recognition rate: ', getRecognitionRate(clf_SVM.predict(testMatrix), testClass))
    print('Logistic Regression recognition rate: ', getRecognitionRate(clf_LR.predict(testMatrix), testClass))
    print('Random Forest recognition rate: ', getRecognitionRate(clf_RF.predict(testMatrix), testClass))
    print('Native Bayes Classifier recognition rate: ', getRecognitionRate(clf_NBC.predict(testMatrix), testClass))
    print('Decision Tree Classifier recognition rate: ', getRecognitionRate(clf_DTC.predict(testMatrix), testClass))
    print('Gradient Boosting Decision Tree recognition rate: ',
          getRecognitionRate(clf_GBDT.predict(testMatrix), testClass))


if __name__ == '__main__':
    print('K个训练集和测试集的平均识别率')
    totalAlgorithm_1()
    print('每类前x%训练，剩余测试，各个模型的识别率')
    totalAlgorithm_2()
    selectRFParam()
    print('随机森林参数调优完成！')

'''
【输出结果】 
K个训练集和测试集的平均识别率 
('K Nearest Neighbor mean recognition rate: ', 0.48914314291650945) 
('Linear Discriminant Analysis mean recognition rate: ', 0.5284076063968655) 
('Support Vector Machine mean recognition rate: ', 0.5271199740575014) 
('Logistic Regression mean recognition rate: ', 0.5620828985391165) 
('Random Forest mean recognition rate: ', 0.512993404168108) 
('Native Bayes Classifier mean recognition rate: ', 0.4467074333715003) 
('Decision Tree Classifier mean recognition rate: ', 0.47351209424438706) 
('Gradient Boosting Decision Tree mean recognition rate: ', 0.5603633086892212) 
每类前x%训练，剩余测试，各个模型的识别率 
('K Nearest Neighbor recognition rate: ', 0.9892818863879957) 
('Linear Discriminant Analysis recognition rate: ', 1.0) 
('Support Vector Machine recognition rate: ', 0.8928188638799571) 
('Logistic Regression recognition rate: ', 0.8494105037513398) 
('Random Forest recognition rate: ', 0.9801714898177921) 
('Native Bayes Classifier recognition rate: ', 0.7604501607717041) 
('Decision Tree Classifier recognition rate: ', 1.0) 
('Gradient Boosting Decision Tree recognition rate: ', 1.0) 
GridSearchCV took 69.51 seconds for 288 candidate parameter settings. 
随机森林参数调优完成！ 
'''
