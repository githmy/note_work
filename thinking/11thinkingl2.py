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
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import jieba
import glob
import json
from subprocess import check_output

# We include the GloVe word vectors in our input files. To include these in your kernel, simple click 'input files' at the top of the notebook, and search 'glove' in the 'datasets' section.

# In[2]:


bpath = os.path.join("..", "data")

TMP_TRAIN_FILE = os.path.join(bpath, "thinking2", "train_compare_origin.csv")
TMP_LABEL_FILE = os.path.join(bpath, "thinking2", "train_compare_label.csv")
TMP_NOLABEL_FILE = os.path.join(bpath, "thinking2", "train_compare_nolabel.csv")
train = pd.read_csv(TMP_TRAIN_FILE, header=0, encoding="utf8", sep=',')
predict_pd = pd.read_csv(TMP_LABEL_FILE, header=0, encoding="utf8", sep=',')
nolabel = pd.read_csv(TMP_NOLABEL_FILE, header=0, encoding="utf8", sep=',')

jieba_userdicts = glob.glob(os.path.join(bpath, "jieba", "*.txt"))
for jieba_userdict in jieba_userdicts:
    jieba.load_userdict(jieba_userdict)
    print("load dict:", jieba_userdicts)

# Set some basic config parameters:

# In[3]:


embed_size = 300  # how big is each word vector
max_features = 20000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100  # max number of words in a comment to use

# Read in our data and replace missing values:

# In[4]:


print(train.head())
list_sentences_train = train["text"].fillna("_na_").values
train = pd.get_dummies(train, columns=['level'])

# # 训练的标签
# yc = np.hstack((ym, yr))
list_classes = [i1 for i1 in train.columns if i1.startswith("level_")]
yl = train[list_classes].values

print(train.head())
# print(yc.shape)
print(yl.shape)

# In[5]:


# 预测的值
# list_sentences_test = predict_pd["Description"].fillna("_na_").values
list_sentences_test_label = predict_pd["text"].fillna("_na_").values
list_sentences_test_nolabel = nolabel["text"].fillna("_na_").values

# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length (with truncation or padding as needed).

# In[6]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test_label = tokenizer.texts_to_sequences(list_sentences_test_label)
list_tokenized_test_nolabel = tokenizer.texts_to_sequences(list_sentences_test_nolabel)
X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te_label = pad_sequences(list_tokenized_test_label, maxlen=maxlen)
X_te_nolabel = pad_sequences(list_tokenized_test_nolabel, maxlen=maxlen)

# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.

# In[7]:


# EMBEDDING_FILE = os.path.join(bpath, "wordvector", "crawl-300d-2M.vec")
EMBEDDING_FILE = os.path.join(bpath, "wordvector", "wiki.zh.vec")


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf-8"))
for o in list(embeddings_index.keys()):
    if len(embeddings_index[o]) != embed_size:
        del embeddings_index[o]

# Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.

# In[8]:


all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

# In[9]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index) + 2)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
embedding_matrix[0] = np.zeros((embed_size))

# Simple bidirectional LSTM with two fully connected layers. We add some dropout to the LSTM since even 2 epochs is enough to overfit.

# In[10]:


inp_l = Input(shape=(maxlen,))
x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp_l)
x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(len(list_classes), activation="softmax")(x)
model_l = Model(inputs=inp_l, outputs=x)
model_l.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Now we're ready to fit out model! Use `validation_split` when not submitting.

# In[11]:


batch_size = 32
epochs = 1000
tensor_path_l = os.path.join(bpath, "logs", "thinking2_l")
model_path_l = os.path.join(bpath, "model", "thinking2_l", "rasa_weights_base.best.hdf5")
checkpoint = ModelCheckpoint(model_path_l, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorb = TensorBoard(log_dir=tensor_path_l, histogram_freq=10, write_graph=True, write_images=True, embeddings_freq=0,
                      embeddings_layer_names=None, embeddings_metadata=None)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
callbacks_list_l = [checkpoint, early, tensorb]

# In[12]:


model_l.fit(X_tr, yl, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list_l)

# And finally, get predictions for the test set and prepare a submission CSV:

# In[23]:


model_l.load_weights(model_path_l)
y_test_l_label = model_l.predict(X_te_label)
y_test_l_nolabel = model_l.predict(X_te_nolabel)

# In[25]:


# sample_submission_label = pd.read_csv(TMP_LABEL_FILE, header=0, encoding="GBK", dtype=str,sep=',')
sample_submission_label = pd.read_csv(TMP_LABEL_FILE, header=0, encoding="utf8", dtype=str, sep=',')
sample_submission_nolabel = pd.read_csv(TMP_NOLABEL_FILE, header=0, encoding="utf8", dtype=str, sep=',')
print(sample_submission_label.columns)
print(sample_submission_nolabel.columns)

sample_submission_label = sample_submission_label[
    ["id", "text", "mainReviewPoints", "reviewPoints", "level", "mainReviewPoints_new"]]
sample_submission_nolabel = sample_submission_nolabel[
    ["id", "text", "mainReviewPoints", "reviewPoints", "level", "mainReviewPoints_new"]]
# sample_submission=sample_submission[["Description", "Level"]]
# 结果转化输出

print(y_test_l_label.shape)

sample_submission_label["level_new"] = np.argmax(y_test_l_label, axis=1)
sample_submission_label = sample_submission_label[
    ["id", "text", "mainReviewPoints", "reviewPoints", "level", "mainReviewPoints_new", "level_new"]]
sample_submission_label.to_csv(TMP_LABEL_FILE, index=False, encoding="utf-8")

print(y_test_l_nolabel.shape)

# for indexs in sample_submission.index:
#     for  i2 in list_classes:  
#         if(sample_submission.loc[indexs,i2] ==sample_submission.loc[indexs,"max"]):
#             sample_submission.loc[indexs,"predict"]=i2
# for i1 in list_classes:
#     sample_submission.rename(columns={i1: "pred_" + i1}, inplace=True)
sample_submission_nolabel["level_new"] = np.argmax(y_test_l_nolabel, axis=1)
sample_submission_nolabel = sample_submission_nolabel[
    ["id", "text", "mainReviewPoints", "reviewPoints", "level", "mainReviewPoints_new", "level_new"]]
sample_submission_nolabel.to_csv(TMP_NOLABEL_FILE, index=False, encoding="utf-8")

print("finish output csv.")

# In[ ]:
print(sample_submission_label.head())
lenth0 = sample_submission_label.shape[0]
print(lenth0)
print(sample_submission_label.loc[sample_submission_label['level'] == sample_submission_label["level_new"]].shape)
print(sample_submission_label[sample_submission_label["level"] == sample_submission_label["level"]].shape)
print(sample_submission_label[
          (sample_submission_label["mainReviewPoints"] == sample_submission_label["mainReviewPoints_new"]) &
          sample_submission_label["mainReviewPoints_new"].notnull()
          & sample_submission_label["mainReviewPoints_new"] == "公式法解一元二次方程"].shape)
lev_c = 0
mai_c = 0
for i1 in sample_submission_label.index:
    if int(sample_submission_label.loc[i1, "level"]) == int(sample_submission_label.loc[i1, "level_new"]):
        lev_c += 1
    if sample_submission_label.loc[i1, "mainReviewPoints"] == sample_submission_label.loc[i1, "mainReviewPoints_new"]:
        mai_c += 1

print(lev_c / lenth0)
print(mai_c / lenth0)
