# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

import numpy as np
import pandas as pd
import tensorflow as tf
import math
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# 构建模型
class TextCNN(object):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        l2_loss = tf.constant(0.0)

        # embedding层
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                 name='W', trainable=True)
            # [batch_size,sequence_length,embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # [batch_size,sequence_length,embedding_size,1]
            # 为了将其应用于conv2d，故需要维度类似于图片，即[batch_size,height,width,channels]
            # 最后的维度1就是channels
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        # 卷积和池化层(包含len(filter_sizes)个)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # [filter_height,filter_width,filter,in_channels,out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # [batch_size,sequence_length-filter_size+1,1,num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # [batch_size,sequence_length-filter_size+1,1,num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # [batch_size,1,1,num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # 合并所有pool的输出
        num_filters_total = num_filters * len(filter_sizes)
        # [batch_size,1,1,num_filter*len(filter_sizes)]
        self.h_pool = tf.concat(pooled_outputs, len(filter_sizes))
        # [bathc_size, num_filter*len(filter_sizes)]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 输出分类
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 计算loss
        with tf.name_scope("loss"):
            # loss
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # 正则化后的loss
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# 用于产生batch
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size  # 每个epoch中包含的batch数量
    for epoch in range(num_epochs):
        # 每个epoch是否进行shuflle
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch + 1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def main():
    # 数据读取
    train = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", quoting=3)

    # 参数
    max_features = 10000  # vocabulary的大小
    maxlen = 500
    embedding_size = 128
    batch_size = 512  # 每个batch中样本的数量
    num_epochs = 20
    max_learning_rate = 0.005
    min_learning_rate = 0.0001
    decay_coefficient = 2.5  # learning_rate的衰减系数
    dropout_keep_prob = 0.5  # dropout的比例
    evaluate_every = 100  # 每100step进行一次eval
    num_filters = 128  # filter的数量
    filter_sizes = [3, 4, 5]

    # 数据处理
    # 建立tokenizer
    tokenizer = Tokenizer(num_words=max_features, lower=True)
    tokenizer.fit_on_texts(list(train['review']) + list(test['review']))
    # word_index = tokenizer.word_index
    x_train = tokenizer.texts_to_sequences(list(train['review']))
    x_train = pad_sequences(x_train, maxlen=maxlen)  # padding
    y_train = to_categorical(list(train['sentiment']))  # one-hot
    x_test = tokenizer.texts_to_sequences(list(test['review']))
    x_test = pad_sequences(x_test, maxlen=maxlen)  # padding
    # 划分训练和验证集
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

    # 模型训练
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,  # 如果指定的设备不存在，允许tf自动分配设备
            log_device_placement=False)  # 不打印设备分配日志
        sess = tf.Session(config=session_conf)  # 使用session_conf对session进行配置
        # 构建模型
        nn = TextCNN(sequence_length=x_train.shape[1],
                     num_classes=y_train.shape[1],
                     vocab_size=max_features,
                     embedding_size=embedding_size,
                     filter_sizes=filter_sizes,
                     num_filters=num_filters)
        # 用于统计全局的step
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(nn.learning_rate)
        tvars = tf.trainable_variables()  # 返回需要训练的variable
        # tf.gradients(nn.loss, tvars)，计算loss对tvars的梯度
        grads, _ = tf.clip_by_global_norm(tf.gradients(nn.loss, tvars), 5)  # 为了防止梯度爆炸，对梯度进行控制
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        batches = batch_iter(np.hstack((x_train, y_train)), batch_size, num_epochs)
        decay_speed = decay_coefficient * len(y_train) / batch_size
        counter = 0  # 用于记录当前的batch数
        for batch in batches:
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(
                -counter / decay_speed)
            counter += 1
            x_batch, y_batch = batch[:, :-2], batch[:, -2:]
            # 训练
            feed_dict = {nn.input_x: x_batch,
                         nn.input_y: y_batch,
                         nn.dropout_keep_prob: dropout_keep_prob,
                         nn.learning_rate: learning_rate}
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, nn.loss, nn.accuracy],
                feed_dict)
            current_step = tf.train.global_step(sess, global_step)
            # Evaluate
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                feed_dict = {
                    nn.input_x: x_dev,
                    nn.input_y: y_dev,
                    nn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, nn.loss, nn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                print("")

        # predict test set
        all_predictions = []
        test_batches = batch_iter(x_test, batch_size, num_epochs=1, shuffle=False)
        for batch in test_batches:
            feed_dict = {
                nn.input_x: batch,
                nn.dropout_keep_prob: 1.0
            }
            predictions = sess.run([nn.predictions], feed_dict)[0]
            all_predictions.extend(list(predictions))


if __name__ == '__main__':
    main()
