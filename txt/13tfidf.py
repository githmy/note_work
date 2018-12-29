# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import jieba

# 1. 原生
tfidf_vectorizer = TfidfVectorizer()
# 2. 自定义
vocabulary = {'丈夫': 0, '买车': 1, '借名': 2, '偿还': 3, '分割': 4, '夫妻': 5, '妻子': 6, '抢劫': 7, '支持': 8, '离婚': 9, '赌债': 10}
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)

# 3. 文字处理
real_test_raw = ['丈夫他抢劫杀人罪了跑路的话要判多少年', '丈夫借名买车离婚时能否要求分割', '妻子离婚时丈夫欠的赌债是否要偿还？', '夫妻一方下落不明 离婚请求获支持']
real_documents = []
for item_text in real_test_raw:
    item_str = word_process.word_cut(item_text)
    # seg_list = jieba.cut(str(review_content), cut_all=False)
    # word_list = [item for item in seg_list if len(item) > 1]
    # text_list.append(list(set(word_list) - set(stop_words)))
    real_documents.append(item_str)



real_vec = tfidf_vectorizer.fit_transform(real_documents)
print(tfidf_vectorizer.idf_)  # 特征对应的权重
print(tfidf_vectorizer.get_feature_names())  # 特征词
print(real_vec.toarray())  # 上面四句话对应的向量表示
