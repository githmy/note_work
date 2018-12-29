def mla_text():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    class TextClassifier():
        def __init__(self, classifier=MultinomialNB()):
            self.classifier = classifier
            self.vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 3), max_features=20000)

        def features(self, X):
            return self.vectorizer.transform(X)

        def fit(self, X, y):
            self.vectorizer.fit(X)
            self.classifier.fit(self.features(X), y)

        def predict(self, x):
            return self.classifier.predict(self.features([x]))

        def score(self, X, y):
            self.classifier.score(self.features(X), y)

    x_train = ["金山毒 专家建议", "解释 战友 很快明白 航母 机电 空调 系统 重要性 阮万林 竖起 大拇指", "实施 共享 停车 问题重重"]
    y_train = ["technology", "military", "car"]
    text_classifier = TextClassifier()
    text_classifier.fit(x_train, y_train)
    print(text_classifier.predict("这 是 有史以来 最 大 的 一次 车舰 演习"))
    print(text_classifier.score(x_test, y_test))


def fasttext():
    # pip3 install cython
    # pip3 install fasttext
    import fasttext
    help(fasttext)
    # 训练
    classifier = fasttext.load_model("classifier.model")
    model = fasttext.cbow("unsupervise_train_data.txt", "model")
    model = fasttext.skipgram("unsupervise_train_data.txt", "model")
    model.words
    model["赛季"]
    # 文本格式 "__label__n , *"
    classifier = fasttext.supervised("train_data.txt", "classifier.model", label_prefix="__label__")
    # 测试
    result = classifier.test("train_data.txt")
    print(result.precision)
    print(result.recall)
    print("num of examples:", result.nexamples)
    # 实际预测
    labels_to_cate = {1: "technology", 2: "car", 3: "entertrainment", 4: "military"}
    texts = ["中新网 日电 2018 预赛 亚洲区 强赛 中国队 韩国队 较量 比赛 上半场 分钟"]
    labels = classifier.predict(texts, k=3)
    print(labels)
    print(labels_to_cate[int(labels[0][0])])
    labels = classifier.predict_proba(texts, k=3)
    print(labels)



if __name__ == "__main__":
    mla_text()
    fasttext()
