# -*- coding: utf-8 -*-
"""
@autor:zjt
模型训练阶段，对比不同模型效果，保存最优模型。
评价模型效果是在测试机上的表现，比较F1值的大小。

"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from preprocess_pack.chara_project import Tfidf_charaselect
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib as jl
from preprocess_pack.get_basic import json_add, get_words_labels
import os
import re
import pickle as pk

models = {'randomforestclassifier': RandomForestClassifier(n_estimators=10), 'naivebayes': GaussianNB(), \
          'svm': SVC(kernel='linear', probability=True, random_state=24), 'LR': LogisticRegression(penalty='l2'), \
          'neural_network': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)}


def process_tain_nlp(allpaths, standjson, data_cut):
    # 1. 功能选择
    model_dkey = "modelpath"
    if "models" not in standjson["purpose"].keys():
        standjson["purpose"]["models"] = {}

    # 2. 模型训练
    n_feature = standjson["purpose"]["chara"]
    words_cols_list = get_words_labels(data_cut.columns, standjson)
    for [wordres, labelname] in words_cols_list:
        tfidf_data, tfidf_model, sk_model = Tfidf_charaselect(data_cut, wordres + "__cut_", labelname, n_feature)
        # print(tfidf_data.shape)
        modelpath = allpaths[model_dkey]
        model_name, model = train_model(models, tfidf_data, data_cut[labelname])
        # print("model_name", model_name)

        tfidf_model_path = os.path.join(modelpath, '%s__%s__%s__tfidf_model.m' % (standjson["json_file"], wordres,
                                                                                  labelname))
        sk_model_path = os.path.join(modelpath,
                                     '%s__%s__%s__sk_model.m' % (standjson["json_file"], wordres, labelname))
        optimal_model_path = os.path.join(modelpath, '%s__%s__%s__optimal_model.m' % (standjson["json_file"], wordres,
                                                                                      labelname))

        jl.dump(tfidf_model, tfidf_model_path)  # tfidf的训练模型保存
        jl.dump(sk_model, sk_model_path)  # 卡方检验选择特征模型保存
        jl.dump(model, optimal_model_path)  # 最优模型保存
        # 3. json补足
        tmpjson = {
            "%s__%s__%s__tfidf" % (standjson["json_file"], wordres, labelname): tfidf_model_path,
            "%s__%s__%s__chi2" % (standjson["json_file"], wordres, labelname): sk_model_path,
            "%s__%s__%s__optimal" % (standjson["json_file"], wordres, labelname): optimal_model_path,
        }
        standjson["purpose"]["models"] = json_add(standjson["purpose"]["models"], tmpjson)
    return standjson


def train_model(models, tfidf_data, labels, tb=0.3):
    """
    训练模型
    :param models: 所需训练的模型种类，字典形式
    :param tfidf_data: 数据框格式，文本转换成tfidf的数据框
    :param labels: 数组格式，和tfidf_data索引相对应
    :param tb:小于1的数据，测试集的数据
    :return: 返回训练的好的模型名称、模型
    """
    # 划分测试机和训练集
    feature_train, feature_test, labes_train, labels_test = train_test_split(tfidf_data, labels, test_size=tb)
    model_f1 = {}
    model_select = {}
    for name, model in models.items():
        model.fit(feature_train.toarray(), labes_train)
        predict_label = model.predict(feature_test.toarray())
        f1 = f1_score(labels_test, predict_label, average='micro')
        model_f1[name] = f1
        model_select[name] = model
    most_model = sorted(model_f1.items(), key=lambda d: d[1], reverse=True)[0][0]

    return most_model, model_select[most_model]
