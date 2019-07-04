# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc
import os
import datetime
import jieba
import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.sdata_helper import batch_iter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# 构建模型
class TextCNN(object):
    def __init__(self, env_conf, model_json):
        self._env_conf = env_conf
        self._model_json = model_json
        # 参数转内部
        self.max_features = self._model_json["max_features"]
        self.maxlen = self._model_json["maxlen"]
        self.embedding_size = self._model_json["embedding_size"]
        self.batch_size = self._model_json["batch_size"]
        self.num_epochs = self._model_json["num_epochs"]
        self.max_learning_rate = self._model_json["max_learning_rate"]
        self.min_learning_rate = self._model_json["min_learning_rate"]
        self.decay_coefficient = self._model_json["decay_coefficient"]
        self.dropout_keep_prob_outnn = self._model_json["dropout_keep_prob"]
        self.evaluate_every = self._model_json["evaluate_every"]
        self.early_stop = self._model_json["early_stop"]
        self.save_step = self._model_json["save_step"]
        self.num_filters = self._model_json["num_filters"]
        self.filter_sizes = self._model_json["filter_sizes"]
        self.l2_reg_lambda = self._model_json["l2_reg_lambda"]
        self.model_name = self._model_json["model_name"]
        self.default_g = None

    def build(self):
        with tf.Graph().as_default():
            l2_loss_l = tf.constant(0.0)
            l2_loss_m = tf.constant(0.0)
            l2_loss_r = tf.constant(0.0)
            l2_loss_sig_m = tf.constant(0.0)
            self.default_g = l2_loss_l.graph
            # 1. 输入层
            with tf.name_scope('input'):
                # sequence_length 即 训练文本的单行长度
                self.input_x = tf.placeholder(tf.int32, [None, self.length_y], name='input_x')
                self.input_y_l = tf.placeholder(tf.float32, [None, self.label_length_l], name='input_y_l')
                self.input_y_r = tf.placeholder(tf.float32, [None, self.label_length_r], name='input_y_r')
                self.input_y_m = tf.placeholder(tf.float32, [None, self.label_length_m], name='input_y_m')
                self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
                self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

            # 2. embedding层
            with tf.name_scope('embedding'):
                # vocab_size 即 max_features
                self.W = tf.Variable(tf.random_uniform([self.max_features, self.embedding_size], -1.0, 1.0),
                                     name='W', trainable=True)
                # [batch_size,sequence_length,embedding_size]
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                # [batch_size,sequence_length,embedding_size,1]
                # 为了将其应用于conv2d，故需要维度类似于图片，即[batch_size,height,width,channels]
                # 最后的维度1就是channels
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            pooled_outputs = []
            # 3. 卷积和池化层(包含len(filter_sizes)个)
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    # [filter_height,filter_width,filter,in_channels,out_channels]
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
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
                    # sequence_length 即 self.length_y
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.length_y - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # 4. 合并所有pool的输出
            num_filters_total = self.num_filters * len(self.filter_sizes)
            # [batch_size,1,1,num_filter*len(filter_sizes)]
            self.h_pool = tf.concat(pooled_outputs, len(self.filter_sizes))
            # [bathc_size, num_filter*len(filter_sizes)]
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # 5. Dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # 6. 输出分类
            with tf.name_scope("output"):
                W_l = tf.get_variable(
                    "W_l",
                    shape=[num_filters_total, self.label_length_l],
                    initializer=tf.contrib.layers.xavier_initializer())
                b_l = tf.Variable(tf.constant(0.1, shape=[self.label_length_l]), name="b_l")
                l2_loss_l += tf.nn.l2_loss(W_l)
                l2_loss_l += tf.nn.l2_loss(b_l)
                self.scores_l = tf.nn.xw_plus_b(self.h_drop, W_l, b_l, name="scores_l")
                self.predictions_l = tf.argmax(self.scores_l, 1, name="predictions_l")
                tf.summary.histogram('predictions_l', self.predictions_l)

                W_r = tf.get_variable(
                    "W_r",
                    shape=[num_filters_total, self.label_length_r],
                    initializer=tf.contrib.layers.xavier_initializer())
                b_r = tf.Variable(tf.constant(0.1, shape=[self.label_length_r]), name="b_r")
                l2_loss_r += tf.nn.l2_loss(W_r)
                l2_loss_r += tf.nn.l2_loss(b_r)
                self.scores_r = tf.nn.xw_plus_b(self.h_drop, W_r, b_r, name="scores_r")
                self.predictions_r = tf.round(tf.nn.sigmoid(self.scores_r))
                tf.summary.histogram('scores_r', self.scores_r)
                tf.summary.histogram('predictions_m', self.predictions_r)

                W_sig_m = tf.get_variable(
                    "W_sig_m",
                    shape=[num_filters_total, self.label_length_m],
                    initializer=tf.contrib.layers.xavier_initializer())
                b_sig_m = tf.Variable(tf.constant(0.1, shape=[self.label_length_m]), name="b_sig_m")
                l2_loss_sig_m += tf.nn.l2_loss(W_sig_m)
                l2_loss_sig_m += tf.nn.l2_loss(b_sig_m)
                self.scores_m = tf.nn.xw_plus_b(self.h_drop, W_sig_m, b_sig_m, name="scores_m")
                self.predictions_m = tf.round(
                    tf.nn.sigmoid(tf.multiply(self.predictions_r, tf.nn.sigmoid(self.scores_m))))
                tf.summary.histogram('scores_m', self.scores_m)
                tf.summary.histogram('predictions_m', self.predictions_m)

            # 7. 计算loss
            with tf.name_scope("loss"):
                # loss
                losses_l = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_l, labels=self.input_y_l)
                # 正则化后的loss
                self.loss_l = tf.reduce_mean(losses_l) + self.l2_reg_lambda * l2_loss_l
                tf.summary.scalar('loss_l', self.loss_l)

                losses_r = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores_r, labels=self.input_y_r)
                self.loss_r = tf.reduce_mean(losses_r) + self.l2_reg_lambda * l2_loss_r
                tf.summary.scalar('loss_r', self.loss_r)

                losses_m = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores_m, labels=self.input_y_m)
                self.loss_m = tf.reduce_mean(losses_m) + self.l2_reg_lambda * l2_loss_m
                tf.summary.scalar('loss_m', self.loss_m)

            # 8. Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions_l = tf.equal(self.predictions_l, tf.argmax(self.input_y_l, 1))
                self.accuracy_l = tf.reduce_mean(tf.cast(correct_predictions_l, "float"), name="accuracy_l")
                tf.summary.scalar('accuracy_l', self.accuracy_l)

                tf_dim_in_r = tf.reduce_sum(tf.abs(self.predictions_r - self.input_y_r), 1)
                correct_predictions_r = tf.less(tf_dim_in_r, tf.ones_like(tf.reduce_sum(self.input_y_r, 1) * 1e-4))
                self.accuracy_r = tf.reduce_mean(tf.cast(correct_predictions_r, tf.float32), name="accuracy_r")
                tf.summary.scalar('accuracy_r', self.accuracy_r)

                tf_dim_in_m = tf.reduce_sum(tf.abs(self.predictions_m - self.input_y_m), 1)
                correct_predictions_m = tf.less(tf_dim_in_m, tf.ones_like(tf.reduce_sum(self.input_y_m, 1) * 1e-4))
                self.accuracy_m = tf.reduce_mean(tf.cast(correct_predictions_m, tf.float32), name="accuracy_m")
                tf.summary.scalar('accuracy_m', self.accuracy_m)

                # self.lr = tf.Variable(self.learning_rate, name="lr")
                # # self.lr = tf.add(self.learning_rate, tf.constant(0.0), name='lr')
                # tf.summary.scalar('lr', self.lr)

    def fit(self, x_train, x_dev, y_train_m, y_dev_m, y_train_r, y_dev_r, y_train_l, y_dev_l):
        bpath = os.path.join("..", "data")
        if self.model_name is None:
            self.model_name = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.model_dir = os.path.join(bpath, "thinking2", "models", self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_name = os.path.join(self.model_dir, "{}.ckpt".format(self.model_name))
        log_dir = os.path.join(bpath, "thinking2", "logs", self.model_name)
        best_loss = 1e9
        best_counter = 0
        # 模型训练
        with self.default_g.as_default():
            # 1. 模型会话
            session_conf = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True),
                allow_soft_placement=True,  # 如果指定的设备不存在，允许tf自动分配设备
                log_device_placement=False)  # 不打印设备分配日志
            # sess = tf.Session(config=session_conf)  # 使用session_conf对session进行配置
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.early_stop)
            with tf.Session(config=session_conf) as sess:
                # 2. 用于统计全局的step
                print("start".center(20, " ").center(80, "*"))
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                tvars = tf.trainable_variables()  # 返回需要训练的variable
                # tf.gradients(nn.loss, tvars)，计算loss对tvars的梯度
                # ttt = tf.gradients([self.loss_l, self.loss_m, self.loss_r], tvars)
                # 为了防止梯度爆炸，对梯度进行控制
                grads, _ = tf.clip_by_global_norm(tf.gradients([self.loss_l, self.loss_m, self.loss_r], tvars), 5)
                grads_and_vars = tuple(zip(grads, tvars))
                # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # 自动更新global_step

                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 保存位置
                sess.run(tf.global_variables_initializer())
                batches = batch_iter(np.hstack((x_train, y_train_l, y_train_m, y_train_r)), self.batch_size,
                                     self.num_epochs)
                decay_speed = self.decay_coefficient * len(y_train_l) / self.batch_size
                counter = 0  # 用于记录当前的batch数
                for batch in batches:
                    learning_rate = self.min_learning_rate + (
                                                                 self.max_learning_rate - self.min_learning_rate) * math.exp(
                        -counter / decay_speed)
                    counter += 1
                    x_batch = batch[:, :self.length_y]
                    y_batch_l = batch[:, self.length_y:self.length_y + self.label_length_l]
                    y_batch_m = batch[:,
                                self.length_y + self.label_length_l:self.length_y + self.label_length_l + self.label_length_m]
                    y_batch_r = batch[:, self.length_y + self.label_length_l + self.label_length_m:]
                    # 训练
                    feed_dict = {self.input_x: x_batch,
                                 self.input_y_l: y_batch_l,
                                 self.input_y_m: y_batch_m,
                                 self.input_y_r: y_batch_r,
                                 self.dropout_keep_prob: self.dropout_keep_prob_outnn,
                                 self.learning_rate: learning_rate}
                    _, step, loss_l, loss_m, loss_r, accuracy_l, accuracy_m, accuracy_r = sess.run(
                        [train_op, global_step, self.loss_l, self.loss_m, self.loss_r, self.accuracy_l, self.accuracy_m,
                         self.accuracy_r], feed_dict)
                    current_step = tf.train.global_step(sess, global_step)
                    # Evaluate
                    if current_step % self.evaluate_every == 0:
                        print("Evaluation:")
                        feed_dict = {
                            self.input_x: x_dev,
                            self.input_y_l: y_dev_l,
                            self.input_y_m: y_dev_m,
                            self.input_y_r: y_dev_r,
                            self.dropout_keep_prob: 1.0
                        }
                        summary, step, loss_l, loss_m, loss_r, accuracy_l, accuracy_m, accuracy_r = sess.run(
                            [merged, global_step, self.loss_l, self.loss_m, self.loss_r, self.accuracy_l,
                             self.accuracy_m, self.accuracy_r], feed_dict)
                        train_writer.add_summary(summary, step)
                        time_str = datetime.datetime.now().isoformat()
                        print(
                            "{}: step {}, loss_l {:g}, loss_m {:g}, loss_r {:g}, acc_l {:g}, acc_m {:g}, acc_r {:g}, lr {:g}".format(
                                time_str, step, loss_l, loss_m, loss_r, accuracy_l, accuracy_m, accuracy_r,
                                learning_rate))
                        # 每10步保存一次参数
                        if step % self.save_step == 0:
                            print("保存模型：", saver.save(sess, model_name, step))
                        if best_loss > loss_m:
                            best_loss = loss_m
                            best_counter = 0
                        else:
                            best_counter += 1
                            if self.early_stop < best_counter:
                                print("保存模型：", saver.save(sess, model_name, step))
                                print("training finished!")
                                return 0
                        print("best_loss: %s, best_counter: %s" % (best_loss, best_counter))

    def predict(self, x_test):
        # 模型训练
        with self.default_g.as_default():
            # 1. 模型会话
            session_conf = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True),
                allow_soft_placement=True,  # 如果指定的设备不存在，允许tf自动分配设备
                log_device_placement=False)  # 不打印设备分配日志
            # sess = tf.Session(config=session_conf)  # 使用session_conf对session进行配置
            with tf.Session(config=session_conf) as sess:
                # predict test set
                all_predictions = []
                test_batches = batch_iter(x_test, self.batch_size, num_epochs=1, shuffle=False)
                for batch in test_batches:
                    feed_dict = {
                        self.input_x: batch,
                        self.dropout_keep_prob: 1.0
                    }
                    predictions = sess.run([self.predictions_l], feed_dict)[0]
                    all_predictions.extend(list(predictions))

    def load_mode(self, modelname):
        pass
        # headname = os.path.join(self.model_dir,self.model_name)
        # saver = tf.train.import_meta_graph('{}.meta'.format(headname))
        # saver.restore(tf.get_default_session(), 'save/filename.ckpt-16000')
        #
        # saver = tf.train.Saver()
        # with tf.Session() as sess:
        #     latest_ckpt = tf.train.latest_checkpoint(self.model_dir)
        #     if latest_ckpt:
        #         """Load model from a checkpoint."""
        #         try:
        #             saver.restore(sess, latest_ckpt)
        #         except tf.errors.NotFoundError as e:
        #             print("Can't load checkpoint")

    def data4train(self):
        # 1. 读取数据源
        self._get_standard_data()
        # 2. 清洗数据
        self._data_clean()
        # 3 数据转token
        self._data_tokenize()
        # 4 标签处理
        self._data_label()
        # 5. 数据切分
        self._data_split()
        return self.x_test, self.x_train, self.x_dev, self.y_train_m, self.y_dev_m, self.y_train_r, self.y_dev_r, self.y_train_l, self.y_dev_l

    def _get_standard_data(self):
        # 内测版读文件，生产调用接口
        if os.getenv('prtest') is None:
            pass
        else:
            # 1.1 数据读取
            bpath = os.path.join("..", "data")
            # train_file = os.path.join(bpath, "thinking2", "question_obj.csv")
            # test_file = os.path.join(bpath, "thinking2", "predict_obj.csv")
            train_file = os.path.join(bpath, "thinking2", "train_compare_origin.csv")
            test_file = os.path.join(bpath, "thinking2", "train_compare_label.csv")
            dict_file = os.path.join(bpath, "thinking2", "review_obj.csv")
            self.train_data = pd.read_csv(train_file, header=0, delimiter=",")
            self.test_data = pd.read_csv(test_file, header=0, delimiter=",")
            dict_pd = pd.read_csv(dict_file, header=0, delimiter=",")
            self.dict_points = {dict_pd.loc[i1, "_id"]: dict_pd.loc[i1, "name"] for i1 in dict_pd.index}
            self.label_list = [i1 for i1 in self.dict_points]
            self.label_fullname_list = [i1 for i1 in self.dict_points.values()]
            self.label_length_r = self.label_length_m = len(self.label_list)

    def _data_clean(self):
        if os.getenv('prtest') is None:
            self.test_data.rename(columns={"description": "text"}, inplace=True)
            self.train_data['mainReviewPoints'] = data_mongo_clean(self.train_data['mainReviewPoints'])
            self.train_data['reviewPoints'] = data_mongo_clean(self.train_data['reviewPoints'])
        else:
            pass

        def pree(strdata):
            return " ".join(jieba.cut(strdata))

        self.train_data["text"] = self.train_data["text"].map(pree)
        self.test_data["text"] = self.test_data["text"].map(pree)

    def _data_tokenize(self):
        # 建立tokenizer
        tokenizer = Tokenizer(num_words=self.max_features, lower=True)
        tokenizer.fit_on_texts(list(self.train_data['text'].values) + list(self.test_data['text'].values))
        # word_index = tokenizer.word_index
        x_train = tokenizer.texts_to_sequences(list(self.train_data['text'].values))
        self.x_train = pad_sequences(x_train, maxlen=self.maxlen)  # padding
        x_test = tokenizer.texts_to_sequences(list(self.test_data['text'].values))
        self.x_test = pad_sequences(x_test, maxlen=self.maxlen)  # padding
        # 长度定义
        self.length_x = self.train_data.shape[0]
        self.length_y = self.maxlen

    def _data_label(self):
        # y_train = to_categorical(list(train['sentiment']))  # one-hot
        self.train_data = pd.get_dummies(self.train_data, columns=['level'])
        self.list_classes = [i1 for i1 in self.train_data.columns if i1.startswith("level_")]
        self.label_length_l = len(self.list_classes)
        self.y_train_l = self.train_data[self.list_classes].values
        self.y_train_r = np.zeros((self.length_x, self.label_length_r), dtype=int)
        self.y_train_m = np.zeros((self.length_x, self.label_length_m), dtype=int)
        for i1 in range(self.length_x):
            try:
                for i2 in self.train_data.loc[i1, "mainReviewPoints"].split(","):
                    # y_train_r[i1, label_list.index(i2)] = 1
                    self.y_train_m[i1, self.label_fullname_list.index(i2)] = 1
            except Exception as e:
                pass
            try:
                for i2 in self.train_data.loc[i1, "reviewPoints"].split(","):
                    # y_train_r[i1, label_list.index(i2)] = 1
                    self.y_train_r[i1, self.label_fullname_list.index(i2)] = 1
            except Exception as e:
                pass

    def _data_split(self):
        # 1.6 划分训练和验证集
        # x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
        self.x_train, self.x_dev, self.y_train_m, self.y_dev_m, self.y_train_r, self.y_dev_r, \
        self.y_train_l, self.y_dev_l = train_test_split(self.x_train, self.y_train_m, self.y_train_r, self.y_train_l,
                                                        test_size=0.2, random_state=0)


def data_mongo_clean(pdser):
    # 清洗 csv 列表
    tmplist = []
    for id1, i1 in enumerate(pdser):
        tmplist.append(",".join(i1.lstrip("[").rstrip("]").strip(" ").strip("'").split("', '")))
    return np.array(tmplist)


if __name__ == '__main__':
    pass
