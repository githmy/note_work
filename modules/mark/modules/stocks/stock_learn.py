# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import copy
import numpy as np
import pandas as pd
from time import time
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.utils import shuffle
from scipy import stats
import pydotplus
import pickle
from sklearn.metrics.pairwise import pairwise_distances_argmin
from txt.basic_tensorflow import neurous_network
import warnings
from hmmlearn import hmm
from math import sqrt


# 计算平方误差
def s_error(A, B):
    return sqrt(np.sum((A - B) * (A - B))) / np.sum(B)


class StockLearn:
    def __init__(self):
        cmd_path = os.getcwd()
        self.data_path = os.path.join(cmd_path, "..", "nocode", "customer")
        self.model_path = os.path.join(self.data_path, "model")
        data_pa = os.path.join(self.data_path, "input", "data")
        self.data_path_stock = os.path.join(data_pa, "stock")
        self.file_stock_info = os.path.join(self.data_path_stock, "stock_info.csv")
        self.data_path_recover = os.path.join(data_pa, "recover")
        self.data_path_res = os.path.join(data_pa, "res")
        # self.file_tmp_feature = os.path.join(self.data_path_res, "profit_date.csv")
        self.file_liquids_order = os.path.join(self.data_path_res, "liquids_order.csv")
        self.file_liquids_mount = os.path.join(self.data_path_res, "liquids_mount.csv")
        self.file_profit_date = os.path.join(self.data_path_res, "profit_date.csv")

    def check_confidence(self, y, yhat):
        # 1. 置信区间
        # stats.t.interval(0.99, df=9, loc=meanv, scale=SE)
        return 0

    def forest_learn(self, label_pd, model_name):
        # 1. 数据预处理
        xcol = [i1 for i1 in label_pd.columns if not i1.startswith("ylabel_")]
        ycol = [i1 for i1 in label_pd.columns if i1.startswith("ylabel_")]
        Xt = label_pd[xcol]
        Yt = label_pd[ycol]

        # 2. 模型训练
        clf = RandomForestRegressor(n_estimators=100)
        t0 = time()
        clf.fit(Xt, Yt)
        t1 = time()
        t = t1 - t0
        print('训练耗时随机森林：%d分钟%.3f秒' % (int(t / 60), t - 60 * int(t / 60)))

        # 3. 模型保存
        # s = pickle.dumps(clf)
        # clf = pickle.loads(s)
        joblib.dump(clf, os.path.join(self.model_path, model_name + ".pkl"))

        # # 4. 模型图
        # # 把所有的树都保存到word
        # for i in xrange(len(clf.estimators_)):
        #     tree.export_graphviz(clf.estimators_[i], os.path.join(self.model_path, model_name + '_%d.dot' % i))
        # # 4.2、给定文件名
        # # tree.export_graphviz(model, out_file='iris1.dot')
        # # 4.3、输出为pdf格式
        # dot_data = tree.export_graphviz(clf, out_file=None,
        #                                 filled=True, rounded=True, special_characters=True)
        # # print dot_data
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf(os.path.join(self.model_path, model_name + ".pdf"))
        # f = open(os.path.join(self.model_path, model_name + ".png"), 'wb')
        # f.write(graph.create_png())
        # f.close()
        return 0

    def forest_predict(self, label_pd, model_name):
        # 1. 数据预处理
        xcol = [i1 for i1 in label_pd.columns if not i1.startswith("ylabel_")]
        ycol = [i1 for i1 in label_pd.columns if i1.startswith("ylabel_")]
        Xv = label_pd[xcol]
        Yv = label_pd[ycol]
        # 1. 模型加载
        clf = joblib.load(os.path.join(self.model_path, model_name + ".pkl"))

        # 2. 预测
        yv_hat = clf.predict(Xv)

        # 3. 结果返回
        pvhat = pd.DataFrame(np.array(yv_hat), columns=ycol)
        for i1 in ycol:
            label_pd["predict_" + i1] = np.array(pvhat[i1])
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 0.01
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 0.01
        label_pd["(y-yhat)/(yhat+1)"] = (label_pd["ylabel_p_change"] - label_pd["predict_ylabel_p_change"]) / (
            label_pd["predict_ylabel_p_change"] + 1)
        label_pd["y/(yhat+1)"] = label_pd["ylabel_p_change"] / (label_pd["predict_ylabel_p_change"] + 1)
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 100
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 100
        # 4. 打印统计数据
        pdobj_positive = label_pd[["y/(yhat+1)", "(y-yhat)/(yhat+1)"]][label_pd["predict_ylabel_p_change"] > 0]
        bias_expect = pdobj_positive["(y-yhat)/(yhat+1)"].sum() / pdobj_positive.shape[0]
        ave_expect = pdobj_positive["y/(yhat+1)"].sum() / pdobj_positive.shape[0]
        print('(y-yhat)/(yhat+1) >0 的期望: ', bias_expect)
        print('yhat/(yhat+1) >0 的期望: ', ave_expect)
        return label_pd

    def nforest_predict(self, label_pd, model_name):
        # 1. 数据预处理
        xcol = [i1 for i1 in label_pd.columns if not i1.startswith("ylabel_")]
        ycol = [i1 for i1 in label_pd.columns if i1.startswith("ylabel_")]
        Xv = label_pd[xcol]
        Yv = label_pd[ycol]
        # 1. 模型加载
        clf = joblib.load(os.path.join(self.model_path, model_name + ".pkl"))

        # 2. 预测
        yv_hat = clf.predict(Xv)

        # 3. 结果返回
        pvhat = pd.DataFrame(np.array(yv_hat), columns=ycol)
        for i1 in ycol:
            label_pd["predict_" + i1] = np.array(pvhat[i1])
        return label_pd

    def hmm_learn(self, label_pd, model_name):
        warnings.filterwarnings("ignore")  # hmmlearn(0.2.0) < sklearn(0.18)
        np.set_printoptions(suppress=True)
        # 1. 数据预处理
        xcol = ['open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5',
                'v_ma10', 'v_ma20', 'turnover', 'amplitude']
        Xt = label_pd[xcol]
        # 2. 模型训练
        n = 5
        model = hmm.GaussianHMM(n_components=n, covariance_type='diag', n_iter=2000)
        t0 = time()
        model.fit(Xt)
        t1 = time()
        t = t1 - t0
        print('训练耗时：%d分钟%.3f秒' % (int(t / 60), t - 60 * int(t / 60)))
        # # 3. 查看固有拟合度
        hidden_statesi = model.predict(Xt)
        # 时间序列长度
        print('##隐状态序列：', hidden_statesi)
        ht = model.predict_proba(Xt)
        # # 时间序列长度*状态长度
        print('##隐状态π的序列：', ht)
        print('##发射B的维度：', model.n_features)
        # 状态长度
        print('##估计初始概率π：', model.startprob_)
        model.startprob_ = ht[len(ht) - 1]
        print('##保存初始概率π：', model.startprob_)
        # 状态长度*状态长度
        print('##估计转移概率A：\n', model.transmat_)
        # 状态长度*特征维度
        print('##估计均值：\n', model.means_)
        # 状态长度*特征维度*特征维度
        print('##估计方差：\n', model.covars_)
        # 4. 模型保存
        joblib.dump(model, os.path.join(self.model_path, model_name + ".pkl"))
        return 0

    def hmm_predict(self, label_pd, model_name):
        np.set_printoptions(suppress=True)
        # 1. 数据预处理
        xcol = ['open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5',
                'v_ma10', 'v_ma20', 'turnover', 'amplitude']
        # Xv = label_pd[xcol]
        # 1. 模型加载
        model = joblib.load(os.path.join(self.model_path, model_name + ".pkl"))
        # 2. 预测
        # 2.1 预测标签的个数
        n_samples = label_pd["close"].shape[0] + 1
        # 2.2 预测数据生成
        sample, labels = model.sample(n_samples=n_samples, random_state=0)
        # 样本序列，时间长度*特征维度
        print('predict ..... ')
        print(sample)
        # 隐状态标签，时间长度
        print(labels)
        hidden_statesi = model.predict(sample)
        # 时间序列长度
        print('##隐状态序列：', labels)
        # 时间序列长度*状态长度
        hv = model.predict_proba(sample)
        print('##隐状态π的序列：', hv)
        print('##发射B的维度：', model.n_features)
        # 状态长度
        print('##估计初始概率π：', model.startprob_)
        # 状态长度*状态长度
        print('##估计转移概率A：\n', model.transmat_)
        # 状态长度*特征维度
        print('##估计均值：\n', model.means_)
        # 状态长度*特征维度*特征维度
        print('##估计方差：\n', model.covars_)

        # 3. 结果返回
        pvhat = pd.DataFrame(np.array(sample[1:n_samples, :]), columns=xcol)
        [label_pd.rename(columns={i1: "ylabel_" + i1}, inplace=True) for i1 in label_pd.columns]
        for i1 in xcol:
            label_pd["predict_ylabel_" + i1] = np.array(pvhat[i1])
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 0.01
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 0.01
        label_pd["(y-yhat)/(yhat+1)"] = (label_pd["ylabel_p_change"] - label_pd["predict_ylabel_p_change"]) / (
            label_pd["predict_ylabel_p_change"] + 1)
        label_pd["y/(yhat+1)"] = label_pd["ylabel_p_change"] / (label_pd["predict_ylabel_p_change"] + 1)
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 100
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 100
        # 4. 打印统计数据
        pdobj_positive = label_pd[["y/(yhat+1)", "(y-yhat)/(yhat+1)"]][label_pd["predict_ylabel_p_change"] > 0]
        bias_expect = pdobj_positive["(y-yhat)/(yhat+1)"].sum() / pdobj_positive.shape[0]
        ave_expect = pdobj_positive["y/(yhat+1)"].sum() / pdobj_positive.shape[0]
        print('(y-yhat)/(yhat+1) >0 的期望: ', bias_expect)
        print('yhat/(yhat+1) >0 的期望: ', ave_expect)
        return label_pd

    def nn_learn(self, label_pd, model_name):
        warnings.filterwarnings("ignore")  # hmmlearn(0.2.0) < sklearn(0.18)
        np.set_printoptions(suppress=True)
        # 1. 数据预处理
        # 2. 模型训练
        t0 = time()
        model = neurous_network(label_pd, batch_size=10)
        t1 = time()
        t = t1 - t0
        print('训练耗时：%d分钟%.3f秒' % (int(t / 60), t - 60 * int(t / 60)))
        # # 3. 查看固有拟合度
        # 4. 模型保存
        joblib.dump(model, os.path.join(self.model_path, model_name + ".pkl"))
        return 0

    def nn_predict(self, label_pd, model_name):
        np.set_printoptions(suppress=True)
        # 1. 数据预处理
        xcol = ['open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5',
                'v_ma10', 'v_ma20', 'turnover']
        # Xv = label_pd[xcol]
        # 1. 模型加载
        print(model_name)
        model = joblib.load(os.path.join(self.model_path, model_name + ".pkl"))
        # 2. 预测
        # 2.1 预测标签的个数
        n_samples = label_pd["close"].shape[0] + 1
        # 2.2 预测数据生成
        print(model)
        sample, labels = model.sample(n_samples=n_samples, random_state=0)
        # 样本序列，时间长度*特征维度
        print('predict ..... ')
        print(sample)
        # 隐状态标签，时间长度
        print(labels)
        hidden_statesi = model.predict(sample)
        # 时间序列长度
        print('##隐状态序列：', labels)
        # 时间序列长度*状态长度
        hv = model.predict_proba(sample)
        print('##隐状态π的序列：', hv)
        print('##发射B的维度：', model.n_features)
        # 状态长度
        print('##估计初始概率π：', model.startprob_)
        # 状态长度*状态长度
        print('##估计转移概率A：\n', model.transmat_)
        # 状态长度*特征维度
        print('##估计均值：\n', model.means_)
        # 状态长度*特征维度*特征维度
        print('##估计方差：\n', model.covars_)

        # 3. 结果返回
        pvhat = pd.DataFrame(np.array(sample[1:n_samples, :]), columns=xcol)
        [label_pd.rename(columns={i1: "ylabel_" + i1}, inplace=True) for i1 in label_pd.columns]
        for i1 in xcol:
            label_pd["predict_ylabel_" + i1] = np.array(pvhat[i1])
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 0.01
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 0.01
        label_pd["(y-yhat)/(yhat+1)"] = (label_pd["ylabel_p_change"] - label_pd["predict_ylabel_p_change"]) / (
            label_pd["predict_ylabel_p_change"] + 1)
        label_pd["y/(yhat+1)"] = label_pd["ylabel_p_change"] / (label_pd["predict_ylabel_p_change"] + 1)
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 100
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 100
        # 4. 打印统计数据
        pdobj_positive = label_pd[["y/(yhat+1)", "(y-yhat)/(yhat+1)"]][label_pd["predict_ylabel_p_change"] > 0]
        bias_expect = pdobj_positive["(y-yhat)/(yhat+1)"].sum() / pdobj_positive.shape[0]
        ave_expect = pdobj_positive["y/(yhat+1)"].sum() / pdobj_positive.shape[0]
        print('(y-yhat)/(yhat+1) >0 的期望: ', bias_expect)
        print('yhat/(yhat+1) >0 的期望: ', ave_expect)
        return label_pd

    def deep_learn(self, label_pd, model_name):
        warnings.filterwarnings("ignore")  # hmmlearn(0.2.0) < sklearn(0.18)
        np.set_printoptions(suppress=True)
        # 1. 数据预处理
        # 2. 模型训练
        t0 = time()
        model = neurous_network(label_pd, batch_size=10)
        t1 = time()
        t = t1 - t0
        print('训练耗时：%d分钟%.3f秒' % (int(t / 60), t - 60 * int(t / 60)))
        # # 3. 查看固有拟合度
        # 4. 模型保存
        joblib.dump(model, os.path.join(self.model_path, model_name + ".pkl"))
        return 0

    def deep_predict(self, label_pd, model_name):
        np.set_printoptions(suppress=True)
        # 1. 数据预处理
        xcol = ['open', 'high', 'close', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5',
                'v_ma10', 'v_ma20', 'turnover']
        # Xv = label_pd[xcol]
        # 1. 模型加载
        model = joblib.load(os.path.join(self.model_path, model_name + ".pkl"))
        # 2. 预测
        # 2.1 预测标签的个数
        n_samples = label_pd["close"].shape[0] + 1
        # 2.2 预测数据生成
        sample, labels = model.sample(n_samples=n_samples, random_state=0)
        # 样本序列，时间长度*特征维度
        print('predict ..... ')
        print(sample)
        # 隐状态标签，时间长度
        print(labels)
        hidden_statesi = model.predict(sample)
        # 时间序列长度
        print('##隐状态序列：', labels)
        # 时间序列长度*状态长度
        hv = model.predict_proba(sample)
        print('##隐状态π的序列：', hv)
        print('##发射B的维度：', model.n_features)
        # 状态长度
        print('##估计初始概率π：', model.startprob_)
        # 状态长度*状态长度
        print('##估计转移概率A：\n', model.transmat_)
        # 状态长度*特征维度
        print('##估计均值：\n', model.means_)
        # 状态长度*特征维度*特征维度
        print('##估计方差：\n', model.covars_)

        # 3. 结果返回
        pvhat = pd.DataFrame(np.array(sample[1:n_samples, :]), columns=xcol)
        [label_pd.rename(columns={i1: "ylabel_" + i1}, inplace=True) for i1 in label_pd.columns]
        for i1 in xcol:
            label_pd["predict_ylabel_" + i1] = np.array(pvhat[i1])
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 0.01
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 0.01
        label_pd["(y-yhat)/(yhat+1)"] = (label_pd["ylabel_p_change"] - label_pd["predict_ylabel_p_change"]) / (
            label_pd["predict_ylabel_p_change"] + 1)
        label_pd["y/(yhat+1)"] = label_pd["ylabel_p_change"] / (label_pd["predict_ylabel_p_change"] + 1)
        label_pd["ylabel_p_change"] = label_pd["ylabel_p_change"] * 100
        label_pd["predict_ylabel_p_change"] = label_pd["predict_ylabel_p_change"] * 100
        # 4. 打印统计数据
        pdobj_positive = label_pd[["y/(yhat+1)", "(y-yhat)/(yhat+1)"]][label_pd["predict_ylabel_p_change"] > 0]
        bias_expect = pdobj_positive["(y-yhat)/(yhat+1)"].sum() / pdobj_positive.shape[0]
        ave_expect = pdobj_positive["y/(yhat+1)"].sum() / pdobj_positive.shape[0]
        print('(y-yhat)/(yhat+1) >0 的期望: ', bias_expect)
        print('yhat/(yhat+1) >0 的期望: ', ave_expect)
        return label_pd
