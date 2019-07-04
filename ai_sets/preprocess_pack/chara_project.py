# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# from sklearn.externals import joblib as jl
def Tfidf_charaselect(dataset, wordname, labelname,n_feature=500):
    """
    把分词后的文本转为tfidf向量并特征选择
    :param dataset: 分词后的数据，数据框格式
    :return: 特征选择后的tfidf向量，稀疏矩阵形式；标签列：Series格式
    """
    Vector = TfidfVectorizer()
    Vector_fit = Vector.fit(dataset[wordname])
    tfidf_vec = Vector_fit.transform(dataset[wordname])
    SK = SelectKBest(chi2, k=n_feature).fit(tfidf_vec, dataset[labelname])
    select_feature = SK.transform(tfidf_vec)
    return select_feature, Vector_fit, SK
