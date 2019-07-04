#!/usr/bin/env python
# coding: utf-8

# # Improved LSTM baseline
# 
# This kernel is a somewhat improved version of [Keras - Bidirectional LSTM baseline](https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051) along with some additional documentation of the steps. (NB: this notebook has been re-run on the new test set.)

# In[1]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
# from keras.models import Graph
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import jieba
import glob
import json
from subprocess import check_output

# We include the GloVe word vectors in our input files. To include these in your kernel, simple click 'input files' at the top of the notebook, and search 'glove' in the 'datasets' section.

# In[2]:


bpath = os.path.join("..", "data")
# EMBEDDING_FILE = os.path.join(bpath, "wordvector", "crawl-300d-2M.vec")
EMBEDDING_FILE = os.path.join(bpath, "wordvector", "wiki.zh.vec")
TRAIN_DATA_FILE = os.path.join(bpath, "thinking2", "question_obj.csv")
VALID_DATA_FILE = os.path.join(bpath, "thinking2", "valid_obj.csv")
predict_file = os.path.join(bpath, "thinking2", "predict_obj.csv")

# tmpo_path = os.path.join(bpath, "thinking2", "predict_obj.csv")
# predict_pd = pd.read_csv(tmpo_path, header=0, encoding="utf8", dtype=str,sep='\t')
# tmpo_path = os.path.join(bpath, "thinking2", "question_obj.csv")
# train_pd = pd.read_csv(tmpo_path, header=0, encoding="utf8", dtype=str,sep=',')
tmpo_path = os.path.join(bpath, "thinking2", "review_obj.csv")
dict_pd = pd.read_csv(tmpo_path, header=0, encoding="utf8", dtype=str, sep=',')
label_list = [i1 for i1 in dict_pd["_id"]]
label_lenth = len(label_list)
print("label_lenth: ", label_lenth)

jieba_userdicts = glob.glob(os.path.join(bpath, "jieba", "*.txt"))
for jieba_userdict in jieba_userdicts:
    jieba.load_userdict(jieba_userdict)
    print("load dict:", jieba_userdicts)

# Set some basic config parameters:

# In[3]:


embed_size = 300  # how big is each word vector
max_features = 20000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100  # max number of words in a comment to use

# 使用原始文件 分出验证集
train_all = pd.read_csv(TRAIN_DATA_FILE)
train_all = train_all.sample(frac=1, random_state=998).reset_index(drop=True)

# Read in our data and replace missing values:

# In[4]:


rm_row_list = []
not_in_label_list = []
not_in_item = []
np_list_r = []
np_list_m = []
for i1 in train_all.index:
    # 1. 清理内容，转为数组
    restt_r = train_all.loc[i1, "reviewPoints"].lstrip("[").rstrip("]").strip(" ").strip("'").split("', '")
    restt_m = train_all.loc[i1, "mainReviewPoints"].lstrip("[").rstrip("]").strip(" ").strip("'").split("', '")
    #     print(i1,train.loc[i1, "reviewPoints"],restt)
    # 2. 判断删除项
    listindex_del_r = []
    listindex_del_m = []
    dealstrl_r = []
    dealstrl_m = []
    for id2, i2 in enumerate(restt_r):
        dealstr_r = i2.replace("\n", "").strip(", '").strip("', ")
        if dealstr_r == "" or dealstr_r == " ":
            listindex_del_r.append(id2)
        elif dealstr_r not in label_list:
            listindex_del_r.append(id2)
            #             not_in_label_list.append([train.loc[i1, "id"], dealstr])
            not_in_item.append(train_all.loc[i1, "id"])
        else:
            dealstrl_r.append(dealstr_r)
    for id2, i2 in enumerate(restt_m):
        dealstr_m = i2.replace("\n", "").strip(", '").strip("', ")
        if dealstr_m == "" or dealstr_m == " ":
            listindex_del_m.append(id2)
        elif dealstr_m not in label_list:
            listindex_del_m.append(id2)
            not_in_item.append(train_all.loc[i1, "id"])
        else:
            dealstrl_m.append(dealstr_m)
    # 3. 删除 删除项
    listindex_del_r.reverse()
    for i2 in listindex_del_r:
        restt_r.pop(i2)
    listindex_del_m.reverse()
    for i2 in listindex_del_m:
        restt_m.pop(i2)
    # 4. 判断移除项
    if len(restt_r) == 0 and len(restt_m) == 0:
        rm_row_list.append(i1)
    else:
        np_list_r.append(dealstrl_r)
        np_list_m.append(dealstrl_m)
# print(i1,len(restt))
print("not_in_item length: ", len(set(not_in_item)))
# print(not_in_item)
print("rm_row_list length: ", len(rm_row_list))
# print(rm_row_list)
# print(np_list_r)

epfile = os.path.join("..", "data", "error.json")
with open(epfile, "w", encoding="utf-8") as f:
    json.dump(list(set(not_in_item)), f, ensure_ascii=False)

# In[5]:


map_file = os.path.join(bpath, "thinking2", "review_obj.csv")
map_pd = pd.read_csv(map_file)
map_points = {map_pd.loc[i1, "_id"]: map_pd.loc[i1, "name"] for i1 in map_pd.index}

# In[31]:


train_before_split = train_all.drop(rm_row_list).reset_index(drop=True)
nolabel = train_all.loc[rm_row_list, :].reset_index(drop=True)

# print(train_all.head())
# print(train_all.shape)
# print(nolabel.head())
# print(nolabel.shape)
# print(train_before_split.head())
# print(train_before_split.shape)

# 没有latex数据的内部转化版
# train = train_all
# predict_pd = pd.read_csv(predict_file, header=0, encoding="utf8", dtype=str,sep='\t')
lenth_train = train_before_split.shape[0]
spint = int(0.9 * lenth_train)
train = train_before_split.loc[0:spint, :].reset_index(drop=True)
predict_pd = train_before_split.loc[spint:, :].reset_index(drop=True)

print(train.shape)
print(train.head())
print(predict_pd.shape)
print(predict_pd.head())

np_list_r = []
np_list_m = []
for i1 in train.index:
    # 1. 清理内容，转为数组
    restt_r = train.loc[i1, "reviewPoints"].lstrip("[").rstrip("]").strip(" ").strip("'").split("', '")
    restt_m = train.loc[i1, "mainReviewPoints"].lstrip("[").rstrip("]").strip(" ").strip("'").split("', '")
    #     print(i1,train.loc[i1, "reviewPoints"],restt)
    # 2. 判断删除项
    listindex_del_r = []
    listindex_del_m = []
    dealstrl_r = []
    dealstrl_m = []
    for id2, i2 in enumerate(restt_r):
        dealstr_r = i2.replace("\n", "").strip(", '").strip("', ")
        if dealstr_r == "" or dealstr_r == " ":
            listindex_del_r.append(id2)
        elif dealstr_r not in label_list:
            listindex_del_r.append(id2)
            #             not_in_label_list.append([train.loc[i1, "id"], dealstr])
            not_in_item.append(train.loc[i1, "id"])
        else:
            dealstrl_r.append(dealstr_r)
    for id2, i2 in enumerate(restt_m):
        dealstr_m = i2.replace("\n", "").strip(", '").strip("', ")
        if dealstr_m == "" or dealstr_m == " ":
            listindex_del_m.append(id2)
        elif dealstr_m not in label_list:
            listindex_del_m.append(id2)
            not_in_item.append(train.loc[i1, "id"])
        else:
            dealstrl_m.append(dealstr_m)
    # 3. 删除 删除项
    listindex_del_r.reverse()
    for i2 in listindex_del_r:
        restt_r.pop(i2)
    listindex_del_m.reverse()
    for i2 in listindex_del_m:
        restt_m.pop(i2)
    # 4. 判断移除项
    if len(restt_r) == 0 and len(restt_m) == 0:
        rm_row_list.append(i1)
    else:
        np_list_r.append(dealstrl_r)
        np_list_m.append(dealstrl_m)

print("np_list_r length: ", len(np_list_r))
print("np_list_m length: ", len(np_list_m))

# # 5. 生成标签列
yr = np.zeros((len(np_list_r), len(label_list)))
ym = np.zeros((len(np_list_m), len(label_list)))
for id1, i1 in enumerate(np_list_r):
    for i2 in i1:
        yr[id1, label_list.index(i2)] = 1
for id1, i1 in enumerate(np_list_m):
    for i2 in i1:
        ym[id1, label_list.index(i2)] = 1

np_list_r = []
np_list_m = []
for i1 in predict_pd.index:
    # 1. 清理内容，转为数组
    restt_r = predict_pd.loc[i1, "reviewPoints"].lstrip("[").rstrip("]").strip(" ").strip("'").split("', '")
    restt_m = predict_pd.loc[i1, "mainReviewPoints"].lstrip("[").rstrip("]").strip(" ").strip("'").split("', '")
    #     print(i1,train.loc[i1, "reviewPoints"],restt)
    # 2. 判断删除项
    listindex_del_r = []
    listindex_del_m = []
    dealstrl_r = []
    dealstrl_m = []
    for id2, i2 in enumerate(restt_r):
        dealstr_r = i2.replace("\n", "").strip(", '").strip("', ")
        if dealstr_r == "" or dealstr_r == " ":
            listindex_del_r.append(id2)
        elif dealstr_r not in label_list:
            listindex_del_r.append(id2)
            #             not_in_label_list.append([train.loc[i1, "id"], dealstr])
            not_in_item.append(predict_pd.loc[i1, "id"])
        else:
            dealstrl_r.append(dealstr_r)
    for id2, i2 in enumerate(restt_m):
        dealstr_m = i2.replace("\n", "").strip(", '").strip("', ")
        if dealstr_m == "" or dealstr_m == " ":
            listindex_del_m.append(id2)
        elif dealstr_m not in label_list:
            listindex_del_m.append(id2)
            not_in_item.append(predict_pd.loc[i1, "id"])
        else:
            dealstrl_m.append(dealstr_m)
    # 3. 删除 删除项
    listindex_del_r.reverse()
    for i2 in listindex_del_r:
        restt_r.pop(i2)
    listindex_del_m.reverse()
    for i2 in listindex_del_m:
        restt_m.pop(i2)
    # 4. 判断移除项
    if len(restt_r) == 0 and len(restt_m) == 0:
        rm_row_list.append(i1)
    else:
        np_list_r.append(dealstrl_r)
        np_list_m.append(dealstrl_m)

print("np_list_r length: ", len(np_list_r))
print("np_list_m length: ", len(np_list_m))

# # 5. 生成标签列
yr_pred = np.zeros((len(np_list_r), len(label_list)))
ym_pred = np.zeros((len(np_list_m), len(label_list)))
for id1, i1 in enumerate(np_list_r):
    for i2 in i1:
        yr_pred[id1, label_list.index(i2)] = 1
for id1, i1 in enumerate(np_list_m):
    for i2 in i1:
        ym_pred[id1, label_list.index(i2)] = 1

# In[32]:


# print(map_points)
for i1 in train.index:
    train.loc[i1, "text"] = " ".join(jieba.cut(train.loc[i1, "text"]))
    tmpstr = ",".join([map_points[label_list[id2]] for id2, i2 in enumerate(ym[i1, :]) if i2 > 0.5])
    train.loc[i1, "mainReviewPoints"] = tmpstr
    tmpstr = ",".join([map_points[label_list[id2]] for id2, i2 in enumerate(yr[i1, :]) if i2 > 0.5])
    train.loc[i1, "reviewPoints"] = tmpstr

# 预测的值
for i1 in predict_pd.index:
    predict_pd.loc[i1, "text"] = " ".join(jieba.cut(predict_pd.loc[i1, "text"]))
    tmpstr = ",".join([map_points[label_list[id2]] for id2, i2 in enumerate(ym_pred[i1, :]) if i2 > 0.5])
    predict_pd.loc[i1, "mainReviewPoints"] = tmpstr
    tmpstr = ",".join([map_points[label_list[id2]] for id2, i2 in enumerate(yr_pred[i1, :]) if i2 > 0.5])
    predict_pd.loc[i1, "reviewPoints"] = tmpstr

for i1 in nolabel.index:
    nolabel.loc[i1, "text"] = " ".join(jieba.cut(nolabel.loc[i1, "text"]))

print(train.head())
# 内侧数据不同版本
TMP_TRAIN_FILE = os.path.join(bpath, "thinking2", "train_compare_origin.csv")
TMP_LABEL_FILE = os.path.join(bpath, "thinking2", "train_compare_label.csv")
TMP_NOLABEL_FILE = os.path.join(bpath, "thinking2", "train_compare_nolabel.csv")
train.to_csv(TMP_TRAIN_FILE, index=False)
predict_pd.to_csv(TMP_LABEL_FILE, index=False)
nolabel.to_csv(TMP_NOLABEL_FILE, index=False)

list_sentences_train = train["text"].fillna("_na_").values
train = pd.get_dummies(train, columns=['level'])
# 训练的标签
yc = np.hstack((ym, yr))

list_classes = [i1 for i1 in train.columns if i1.startswith("level_")]
yl = train[list_classes].values

print(yc.shape)
print(yl.shape)

# In[8]:


# for i1 in predict_pd.index:
#     predict_pd.loc[i1, "Description"]=" ".join(jieba.cut(predict_pd.loc[i1, "Description"]))
#     if i1 % 1000==0:
#         print(i1)
print(predict_pd.columns)
# list_sentences_test = predict_pd["Description"].fillna("_na_").values
list_sentences_test_label = predict_pd["text"].fillna("_na_").values
list_sentences_test_nolabel = nolabel["text"].fillna("_na_").values

# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length (with truncation or padding as needed).

# In[9]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test_label = tokenizer.texts_to_sequences(list_sentences_test_label)
list_tokenized_test_nolabel = tokenizer.texts_to_sequences(list_sentences_test_nolabel)
X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te_label = pad_sequences(list_tokenized_test_label, maxlen=maxlen)
X_te_nolabel = pad_sequences(list_tokenized_test_nolabel, maxlen=maxlen)


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.

# In[10]:


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf-8"))
for o in list(embeddings_index.keys()):
    if len(embeddings_index[o]) != embed_size:
        del embeddings_index[o]

# Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.

# In[11]:


all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

# In[12]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index) + 2)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
embedding_matrix[0] = np.zeros((embed_size))

# Simple bidirectional LSTM with two fully connected layers. We add some dropout to the LSTM since even 2 epochs is enough to overfit.

# In[13]:


inp_l = Input(shape=(maxlen,))
x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp_l)
x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(len(list_classes), activation="softmax")(x)
model_l = Model(inputs=inp_l, outputs=x)
model_l.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[14]:


inp_r = Input(shape=(maxlen,))
x_r = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp_r)
x_r = Bidirectional(LSTM(150, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x_r)
x_r = GlobalMaxPool1D()(x_r)
x_r = Dense(512, activation="relu")(x_r)
x_r = Dropout(0.2)(x_r)
x_r = Dense(yr.shape[1], activation="sigmoid")(x_r)
model_r = Model(inputs=inp_r, outputs=x_r)
model_r.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[15]:


inp_m = Input(shape=(maxlen,))
x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp_m)
x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
x = GlobalMaxPool1D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(ym.shape[1], activation="sigmoid")(x)
model_m = Model(inputs=inp_m, outputs=x)
model_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[16]:


inp_c = Input(shape=(maxlen,))
x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp_c)
x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
x = GlobalMaxPool1D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(yc.shape[1], activation="sigmoid")(x)
model_c = Model(inputs=inp_c, outputs=x)
model_c.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Now we're ready to fit out model! Use `validation_split` when not submitting.

# In[17]:


batch_size = 32
epochs = 1000
tensor_path_l = os.path.join(bpath, "logs", "thinking2_l")
model_path_l = os.path.join(bpath, "model", "thinking2_l", "rasa_weights_base.best.hdf5")
checkpoint = ModelCheckpoint(model_path_l, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorb = TensorBoard(log_dir=tensor_path_l, histogram_freq=10, write_graph=True, write_images=True, embeddings_freq=0,
                      embeddings_layer_names=None, embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list_l = [checkpoint, early, tensorb]

# In[18]:


batch_size = 32
epochs = 1000
tensor_path_m = os.path.join(bpath, "logs", "thinking2_m")
model_path_m = os.path.join(bpath, "model", "thinking2_m", "rasa_weights_base.best.hdf5")
checkpoint = ModelCheckpoint(model_path_m, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorb = TensorBoard(log_dir=tensor_path_m, histogram_freq=10, write_graph=True, write_images=True, embeddings_freq=0,
                      embeddings_layer_names=None, embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list_m = [checkpoint, early, tensorb]

# In[19]:


batch_size = 32
epochs = 1000
tensor_path_r = os.path.join(bpath, "logs", "thinking2_r")
model_path_r = os.path.join(bpath, "model", "thinking2_r", "rasa_weights_base.best.hdf5")
checkpoint = ModelCheckpoint(model_path_r, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorb = TensorBoard(log_dir=tensor_path_r, histogram_freq=10, write_graph=True, write_images=True, embeddings_freq=0,
                      embeddings_layer_names=None, embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list_r = [checkpoint, early, tensorb]

# In[20]:


batch_size = 32
epochs = 1000
tensor_path_c = os.path.join(bpath, "logs", "thinking2_c")
model_path_c = os.path.join(bpath, "model", "thinking2_c", "rasa_weights_base.best.hdf5")
checkpoint = ModelCheckpoint(model_path_c, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorb = TensorBoard(log_dir=tensor_path_c, histogram_freq=10, write_graph=True, write_images=True, embeddings_freq=0,
                      embeddings_layer_names=None, embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list_c = [checkpoint, early, tensorb]

# In[21]:


# model_c.fit(X_tr, yc, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list_c)


# In[22]:


model_m.fit(X_tr, ym, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list_m)

# In[23]:


# model_r.fit(X_tr, yr, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list_r)


# In[24]:


# model_l.fit(X_tr, yl, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list_l)


# And finally, get predictions for the test set and prepare a submission CSV:

# In[25]:


# y_test = model.predict([X_te], batch_size=1024, verbose=1)
# sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
# sample_submission[list_classes] = y_test
# sample_submission.to_csv('submission.csv', index=False)


# In[26]:


# model_l.load_weights(model_path_l)
# y_test_l = model_l.predict(X_te)


# In[27]:


# model_r.load_weights(model_path_r)
# y_test_r = model_r.predict(X_te)


# In[33]:


model_m.load_weights(model_path_m)
y_test_m_label = model_m.predict(X_te_label)
y_test_m_nolabel = model_m.predict(X_te_nolabel)

# In[34]:


# model_c.load_weights(model_path_c)
# y_test_c = model_c.predict(X_te)


# In[35]:


sample_submission_label = pd.read_csv(TMP_LABEL_FILE, header=0, encoding="utf8", dtype=str, sep=',')
sample_submission_nolabel = pd.read_csv(TMP_NOLABEL_FILE, header=0, encoding="utf8", dtype=str, sep=',')
print(sample_submission_label.columns)
print(sample_submission_nolabel.columns)
sample_submission_label = sample_submission_label[["id", "text", "mainReviewPoints", "reviewPoints", "level"]]
sample_submission_nolabel = sample_submission_nolabel[["id", "text", "mainReviewPoints", "reviewPoints", "level"]]
# sample_submission=sample_submission[["Description", "Level"]]
# 结果转化输出

reshape = y_test_m_label.shape
print(reshape)
flist_m = []
flist_r = []
for i1 in range(reshape[0]):
    strlist_m = []
    strlist_r = []
    for i2 in range(reshape[1]):
        if y_test_m_label[i1, i2] > 0.5:
            if i2 >= label_lenth:
                strlist_r.append(map_points[label_list[i2 - label_lenth]])
            else:
                strlist_m.append(map_points[label_list[i2]])
    flist_m.append(",".join(strlist_m))
    flist_r.append(",".join(strlist_r))
np_flist_m = np.array(flist_m)
np_flist_r = np.array(flist_r)
sample_submission_label["mainReviewPoints_new"] = np_flist_m
sample_submission_label = sample_submission_label[
    ["id", "text", "mainReviewPoints", "reviewPoints", "level", "mainReviewPoints_new"]]
sample_submission_label.to_csv(TMP_LABEL_FILE, index=False)

reshape = y_test_m_nolabel.shape
print(reshape)
flist_m = []
flist_r = []
for i1 in range(reshape[0]):
    strlist_m = []
    strlist_r = []
    for i2 in range(reshape[1]):
        if y_test_m_nolabel[i1, i2] > 0.5:
            if i2 >= label_lenth:
                strlist_r.append(map_points[label_list[i2 - label_lenth]])
            else:
                strlist_m.append(map_points[label_list[i2]])
    flist_m.append(",".join(strlist_m))
    flist_r.append(",".join(strlist_r))
np_flist_m = np.array(flist_m)
np_flist_r = np.array(flist_r)
sample_submission_nolabel["mainReviewPoints_new"] = np_flist_m
# sample_submission["reviewPoints_new"] = np_flist_r

# for indexs in sample_submission.index:  
#     for  i2 in list_classes:  
#         if(sample_submission.loc[indexs,i2] ==sample_submission.loc[indexs,"max"]):
#             sample_submission.loc[indexs,"predict"]=i2
# for i1 in list_classes:
#     sample_submission.rename(columns={i1: "pred_" + i1}, inplace=True)
sample_submission_nolabel = sample_submission_nolabel[
    ["id", "text", "mainReviewPoints", "reviewPoints", "level", "mainReviewPoints_new"]]
sample_submission_nolabel.to_csv(TMP_NOLABEL_FILE, index=False)
# print(sample_submission["mainReviewPoints"])
# print(sample_submission["mainReviewPoints"][0])
# print(type(sample_submission["mainReviewPoints"][0]))
# print(sample_submission[sample_submission["mainReviewPoints"]!=""])
print("finish output csv.")


# In[ ]:
