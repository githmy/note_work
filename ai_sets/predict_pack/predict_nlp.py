# -*- coding: utf-8 -*-
"""
调用已经训练好的模型对新样本进行预测
"""
import pickle as pk
import sklearn
from sklearn.externals import joblib as jl
from preprocess_pack.get_basic import get_words_labels
import os


def predict_nlp(standjson, pddata):
    """
    读取训练好的模型，对处理好的数据进行预测
    :param path: 模型存在路径
    :param tfidf_data: 处理好的预测数据
    :return: 返回预测label
    """
    data_back = {}
    tmpjson = standjson["purpose"]["models"]
    words_cols_list = get_words_labels(pddata.columns, standjson)
    for [wordres, labelname] in words_cols_list:
        tfidf_model_path = tmpjson["%s__%s__%s__tfidf" % (standjson["json_file"], wordres, labelname)]
        sk_model_path = tmpjson["%s__%s__%s__chi2" % (standjson["json_file"], wordres, labelname)]
        optimal_model_path = tmpjson["%s__%s__%s__optimal" % (standjson["json_file"], wordres, labelname)]
        tfidf = jl.load(tfidf_model_path)
        chi2 = jl.load(sk_model_path)
        model = jl.load(optimal_model_path)
        data_tfidf = tfidf.transform(pddata[wordres + "__cut_"])
        data_feature = chi2.transform(data_tfidf)
        predict_label = model.predict(data_feature)
        data_back[labelname] = predict_label
    return data_back
