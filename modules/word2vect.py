# -*- coding: utf-8 -*-
import word2vec

# 3.2 构建词向量
sentences = "水 里 没有 油"
fname = "a.txt"
model = word2vec.word2vec(sentences, size=100, window=5, min_count=5, workers=4)
model.save(fname)
model = word2vec.word2vec.load(fname)
model.wv['油']

# word2vec.word2vec('corpusSegDone.txt', 'corpusWord2Vec.bin', size=300,verbose=True)
model = word2vec.word2vec.load('corpusWord2Vec.bin')
# 3.3.1 查看词向量
print(model.vectors)
# 3.3.2 查看词表中的词
index = 1000
print(model.vocab[index])
# 3.3.3 显示空间距离相近的词
indexes = model.cosine(u'油')
for index in indexes[0]:
    print(model.vocab[index])

if __name__ == '__main__':
    # 1. 测试
    print(aa)
