import numpy as np
from tensorflow.python.framework import dtypes
import tensorflow as tf
import time
import os
import math
import tensorflow as tf
from utils.log_tool import *
from utils.sdata_helper import batch_iter_list


# This is abstract class. You need to implement yours.
class AbstractModeltensor(object):
    def __init__(self, config=None):
        self.config = config

    # You need to override this method.
    def buildModel(self):
        raise NotImplementedError("You need to implement your own model.")


class CRNN(AbstractModeltensor):
    def __init__(self, ave_list, bband_list, config=None):
        super(CRNN, self).__init__(config)
        self.modeldic = {
            "cnn_dense": self._cnn_dense_model,  # 原始结构
            "cnn_dense_more": self._cnn_dense_more_model,
            "cnn_full": self._cnn_full_model,  # 原始结构
            "full": self._fullmodel,  # 原始结构
            "one": self._one,  # 原始改单层结构
            "one_y": self._one_y,  # 原始改单层结构
            "one_space": self._one_space,  # 原始改单层结构
            "one_attent": self._one_attent,  # 原始改单层结构
            "one_attent60": self._one_attent60,  # 原始改单层结构
        }
        self.ydim = 1
        self.keep_prob_ph = config["dropout"]
        self.base_dim, self.much_dim = len(bband_list) * 3, len(bband_list) * 4
        self.input_dim = len(ave_list) * (2 * len(ave_list) + 2)
        with tf.name_scope('Inputs'):
            self.input_p = tf.placeholder(tf.float32, [None, self.input_dim])
            self.learn_rate_p = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            self.lr_decay = tf.placeholder(dtype=tf.float32, shape=[])
        with tf.name_scope('Outputs'):
            self.target_base_y = tf.placeholder(dtype=tf.float32, shape=[None, self.base_dim])
            self.target_much_y = tf.placeholder(dtype=tf.float32, shape=[None, self.much_dim])

    def buildModel(self):
        # 不同选择加载
        self.modeldic[self.config["modelname"]]()
        # 打印打包
        self.merged = tf.summary.merge_all()
        # 损失目标
        self.train_op = []
        for i2 in self.train_list:
            self.train_op.append(tf.train.AdamOptimizer(self.learn_rate_p).minimize(i2))
        # 同一保存加载
        self.saver = tf.train.Saver(tf.global_variables())
        # return self.saver

    def _cnn_full_model(self):
        # 部分1，预测值
        with tf.variable_scope('lstm1', initializer=tf.random_normal_initializer()):
            self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
            state1 = self.cell1.zero_state(batch_size=self.batch_p, dtype=tf.float32)
            outputs1, last_states1 = tf.nn.dynamic_rnn(cell=self.cell1, dtype=tf.float32,
                                                       # sequence_length=[self.config["scope"]] * self.batch_p,
                                                       inputs=self.input_p)
        with tf.variable_scope('lstm2', initializer=tf.random_normal_initializer()):
            self.cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
            state2 = self.cell2.zero_state(batch_size=self.batch_p, dtype=tf.float32)
            outputs2, last_states2 = tf.nn.dynamic_rnn(cell=self.cell2, dtype=tf.float32,
                                                       # sequence_length=[self.config["scope"]] * self.batch_p,
                                                       inputs=outputs1)
        w1 = tf.Variable(tf.random_normal([128, 1]), name="Wy")
        b1 = tf.Variable(tf.random_normal([1]), name="by")
        outputlink = outputs2[:, -1, :]
        # outputlink = tf.nn.dropout(outputlink, 0.8)
        y = tf.matmul(outputlink, w1) + b1
        # 损失返回值
        y_loss = tf.reduce_mean(tf.square(y - self.target_y), name="y_loss")

        # 部分2，预测操作
        w2 = tf.Variable(tf.random_normal([128, self.config["outspace"]]), name="Ws")
        b2 = tf.Variable(tf.random_normal([self.config["outspace"]]), name="bs")
        # 空间生成
        outspace = tf.add(tf.matmul(outputlink, w2), b2, name="outspace")
        # 空间的选择
        self.index_space = tf.argmax(outspace, axis=-1, output_type=dtypes.int32, name="outspace_id")
        # 空间的选择 损失函数
        space_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outspace, labels=self.space_chice), name="space_loss")

        # 空间对应的奖励 如果对
        self.god_posi = tf.one_hot(indices=self.space_chice, depth=self.config["outspace"], axis=1)
        self.pred_posi = tf.one_hot(indices=self.index_space, depth=self.config["outspace"], axis=1)
        self.space_if_benefit = tf.stack([1 + y[:, 0], y[:, 0] / y[:, 0], 1 - y[:, 0]], axis=-1,
                                         name="space_if_benefit")
        self.god_benefit_list = tf.reduce_sum(self.space_if_benefit * self.god_posi, axis=-1)
        # 猜错的获取 实际盈利值的负数
        norightorzero = tf.logical_and(
            tf.not_equal(self.index_space, self.space_chice), tf.not_equal(self.index_space, 1))
        fail_bene = tf.reduce_sum(tf.boolean_mask(self.god_benefit_list, norightorzero))
        # 猜对的获取 实际盈利值
        yesright = tf.equal(self.index_space, self.space_chice)
        win_bene = tf.reduce_sum(tf.boolean_mask(self.god_benefit_list, yesright))
        # 总盈利：奖励 - 损失值
        pred_bene = tf.subtract(win_bene, fail_bene, name="pred_bene")
        # 假如全盈利：奖励 + 损失值
        god_benefit = tf.reduce_sum(self.god_benefit_list)
        # 百分系数减慢对之前y的影响
        bene_rate = pred_bene / god_benefit * 100
        # self.learn_rate = tf.Variable(self.learn_rate_p, name="lr", trainable=False)
        # self.update_lr = tf.assign(self.learn_rate, tf.multiply(self.lr_decay, self.learn_rate))
        self.train_list = [y_loss, space_loss, -bene_rate]
        self.valid_list = [y, self.index_space, pred_bene, god_benefit, bene_rate]
        self.pred_list = [y, self.index_space]

        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss', y_loss)  # 记录标量的变化
        tf.summary.scalar('pred_bene', pred_bene)  # 记录标量的变化
        tf.summary.scalar('space_loss', space_loss)  # 记录标量的变化
        return None

    def _cnn_dense_model(self):
        # 部分1，预测值
        dense1 = tf.layers.dense(inputs=self.input_p, units=128, activation=tf.nn.relu, name="layer_dense1")
        concat1 = tf.concat([self.input_p, dense1], 1, name='concat1')
        denseo1 = tf.nn.dropout(concat1, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense1', dense1)  # 记录标量的变化
        dense2 = tf.layers.dense(inputs=denseo1, units=512, activation=tf.nn.relu, name="layer_dense2")
        concat2 = tf.concat([self.input_p, dense1, dense2], 1, name='concat2')
        denseo2 = tf.nn.dropout(concat2, keep_prob=self.keep_prob_ph)
        dense3 = tf.layers.dense(inputs=denseo2, units=256, activation=tf.nn.relu, name="layer_dense3")
        denseo3 = tf.nn.dropout(dense3, keep_prob=self.keep_prob_ph)
        dense4 = tf.layers.dense(inputs=denseo3, units=128, activation=tf.nn.relu, name="layer_dense4")
        denseo4 = tf.nn.dropout(dense4, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense2', dense2)  # 记录标量的变化
        y_base = tf.layers.dense(inputs=denseo4, units=self.base_dim, activation=None, name="y_base")
        y_much = tf.layers.dense(inputs=denseo4, units=self.much_dim, activation=None, name="y_much")
        tf.summary.histogram('y_base', y_base)  # 记录标量的变化
        tf.summary.histogram('y_much', y_much)  # 记录标量的变化
        # 损失返回值
        y_loss_base = tf.reduce_mean(tf.square(y_base - self.target_base_y), name="y_loss_base")
        y_loss_much = tf.reduce_mean(tf.square(y_much - self.target_much_y), name="y_loss_much")
        # 猜错的获取 实际盈利值的负数
        # self.learn_rate = tf.Variable(self.learn_rate_p, name="lr", trainable=False)
        # self.update_lr = tf.assign(self.learn_rate, tf.multiply(self.lr_decay, self.learn_rate))
        self.train_list = [y_loss_base, y_loss_much]
        self.valid_list = [y_loss_base, y_loss_much]
        self.pred_list = [y_base, y_much]
        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss_base', y_loss_base)  # 记录标量的变化
        tf.summary.scalar('y_loss_much', y_loss_much)  # 记录标量的变化

    def _cnn_dense_more_model(self):
        # 部分1，预测值
        dense1 = tf.layers.dense(inputs=self.input_p, units=128, activation=tf.nn.relu, name="layer_dense1")
        concat1 = tf.concat([self.input_p, dense1], 1, name='concat1')
        denseo1 = tf.nn.dropout(concat1, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense1', dense1)  # 记录标量的变化
        dense2 = tf.layers.dense(inputs=denseo1, units=512, activation=tf.nn.relu, name="layer_dense2")
        concat2 = tf.concat([self.input_p, dense1, dense2], 1, name='concat2')
        denseo2 = tf.nn.dropout(concat2, keep_prob=self.keep_prob_ph)
        dense3 = tf.layers.dense(inputs=denseo2, units=256, activation=tf.nn.relu, name="layer_dense3")
        concat3 = tf.concat([self.input_p, dense1, dense2, dense3], 1, name='concat3')
        denseo3 = tf.nn.dropout(concat3, keep_prob=self.keep_prob_ph)
        dense4 = tf.layers.dense(inputs=denseo3, units=128, activation=tf.nn.relu, name="layer_dense4")
        denseo4 = tf.nn.dropout(dense4, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense2', dense2)  # 记录标量的变化
        y_base = tf.layers.dense(inputs=denseo4, units=self.base_dim, activation=None, name="y_base")
        y_much = tf.layers.dense(inputs=denseo4, units=self.much_dim, activation=None, name="y_much")
        tf.summary.histogram('y_base', y_base)  # 记录标量的变化
        tf.summary.histogram('y_much', y_much)  # 记录标量的变化
        # 损失返回值
        y_loss_base = tf.reduce_mean(tf.square(y_base - self.target_base_y), name="y_loss_base")
        y_loss_much = tf.reduce_mean(tf.square(y_much - self.target_much_y), name="y_loss_much")
        # 猜错的获取 实际盈利值的负数
        # self.learn_rate = tf.Variable(self.learn_rate_p, name="lr", trainable=False)
        # self.update_lr = tf.assign(self.learn_rate, tf.multiply(self.lr_decay, self.learn_rate))
        self.train_list = [y_loss_base, y_loss_much]
        self.valid_list = [y_loss_base, y_loss_much]
        self.pred_list = [y_base, y_much]
        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss_base', y_loss_base)  # 记录标量的变化
        tf.summary.scalar('y_loss_much', y_loss_much)  # 记录标量的变化

    def _fullmodel(self):
        # 部分1，预测值
        return None

    def _one_attent60(self):
        # 部分1，预测值
        pass

    def _one_attent(self):
        # 部分1，预测值
        pass

    def _one_y(self):
        # 部分1，预测值
        pass

    def _one_space(self):
        # 部分1，预测值
        pass

    def _one(self):
        # 部分1，预测值
        pass

    def batch_train(self, inputs_t, targets_base_t, targets_much_t, inputs_v, targets_base_v, targets_much_v,
                    batch_size=8, num_epochs=1, retrain=True):
        # 设置
        dataiter = batch_iter_list([inputs_t, targets_base_t, targets_much_t], batch_size, num_epochs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if self.config["retrain"] == 1:
                model_dir = os.path.join(model_path, "model_%s" % self.config["tailname"])
                latest_ckpt = tf.train.latest_checkpoint(model_dir)
                if os.path.isfile("{}.index".format(latest_ckpt)):
                    self.saver.restore(sess, latest_ckpt)
                    print("retraining {}".format(latest_ckpt))
                else:
                    sess.run(tf.global_variables_initializer())
                    print("no old model, training new----")
            writer = tf.summary.FileWriter(os.path.join(log_path, "logs_%s" % self.config["tailname"]), sess.graph)
            global_n = 0
            stop_n = 0
            startt = time.time()
            pre_t_base_loss = pre_t_much_loss = pre_v_much_loss = pre_v_base_loss = 100000
            for epoch in range(num_epochs):
                starte = time.time()
                losslist = [0]
                for batch_num in range(inputs_t.shape[0] // batch_size + 1):
                    # 获取数据
                    inputs_x_t, inputs_ybase_t, inputs_ymuch_t = next(dataiter)
                    feed_dict_t = {
                        self.input_p: inputs_x_t,
                        self.target_base_y: inputs_ybase_t,
                        self.target_much_y: inputs_ymuch_t,
                        self.learn_rate_p: self.config["learn_rate"],
                        self.lr_decay: 1,
                    }
                    # 更新学习率
                    # tmplr = sess.run(self.update_lr)
                    for _ in range(self.config["single_num"]):
                        # 更新速度
                        sess.run(self.train_op, feed_dict_t)
                        global_n += 1
                    losslist_t = sess.run(self.train_list, feed_dict_t)
                    result = sess.run(self.merged, feed_dict_t)
                    if batch_num % 20 == 0:
                        writer.add_summary(result, global_n)
                        self.saver.save(sess, os.path.join(model_path, 'model_%s' % self.config["tailname"],
                                                           self.config["modelfile"]), global_step=global_n)
                        print("batch %s, step %s, time: %s s, y_loss_base_t %s, y_loss_much_t %s" % (
                            batch_num, global_n, time.time() - starte, losslist_t[0], losslist_t[1]))
                # valid part
                feed_dict_v = {
                    self.input_p: inputs_v,
                    self.target_base_y: targets_base_v,
                    self.target_much_y: targets_much_v,
                    self.learn_rate_p: self.config["learn_rate"],
                    self.lr_decay: 1,
                }
                losslist_v = sess.run(self.valid_list, feed_dict_v)
                if losslist_t[0] < pre_t_base_loss and losslist_v[0] < pre_v_base_loss:
                    stop_n += 1
                    if stop_n > self.config["early_stop"]:
                        break
                else:
                    stop_n = 0
                print("epoch %s, step %s, stop_n %s, time: %s s, y_loss_base_v %s, y_loss_much_v %s" % (
                    epoch, global_n, stop_n, time.time() - starte, losslist_v[0], losslist_v[1]))
                pre_t_base_loss = losslist_t[0]
                pre_t_much_loss = losslist_t[1]
                pre_v_base_loss = losslist_v[0]
                pre_v_much_loss = losslist_v[1]
            writer.close()
            print("total time: %s s" % (time.time() - startt))
        # 结束
        print("train finished!")
        return None

    def predict(self, inputs):
        # self.base_dim, self.much_dim = base_dim, much_dim
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model_dir = os.path.join(model_path, "model_%s" % self.config["tailname"])
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, latest_ckpt)
            feed_dict = {
                self.input_p: inputs,
            }
            teslist = sess.run(self.pred_list, feed_dict)
            return teslist


class CRNNevery(AbstractModeltensor):
    def __init__(self, ave_list, bband_list, config=None):
        super(CRNNevery, self).__init__(config)
        self.modeldic = {
            "cnn_dense": self._cnn_dense_model,  # 原始结构
            "cnn_dense_more": self._cnn_dense_more_model,
            "cnn_full": self._cnn_full_model,  # 原始结构
        }
        self.ydim = 1
        self.keep_prob_ph = config["dropout"]
        self.out_dim = len(bband_list)
        self.input_dim = len(ave_list) * (2 * len(ave_list) + 2)
        with tf.name_scope('Inputs'):
            self.input_p = tf.placeholder(tf.float32, [None, self.input_dim])
            self.learn_rate_p = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            self.lr_decay = tf.placeholder(dtype=tf.float32, shape=[])
        with tf.name_scope('Outputs'):
            self.target_base_y = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])
            self.reta = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])
            self.reth = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])
            self.retl = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])
            self.stdup = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])
            self.stddw = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])
            self.drawup = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])
            self.drawdw = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim])

    def buildModel(self):
        # 不同选择加载
        self.modeldic[self.config["modelname"]]()
        # 打印打包
        self.merged = tf.summary.merge_all()
        # 损失目标
        self.train_op = []
        for i2 in self.train_list:
            self.train_op.append(tf.train.AdamOptimizer(self.learn_rate_p).minimize(i2))
        # 同一保存加载
        self.saver = tf.train.Saver(tf.global_variables())
        # return self.saver

    def _cnn_full_model(self):
        # 部分1，预测值
        with tf.variable_scope('lstm1', initializer=tf.random_normal_initializer()):
            self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
            state1 = self.cell1.zero_state(batch_size=self.batch_p, dtype=tf.float32)
            outputs1, last_states1 = tf.nn.dynamic_rnn(cell=self.cell1, dtype=tf.float32,
                                                       # sequence_length=[self.config["scope"]] * self.batch_p,
                                                       inputs=self.input_p)
        with tf.variable_scope('lstm2', initializer=tf.random_normal_initializer()):
            self.cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
            state2 = self.cell2.zero_state(batch_size=self.batch_p, dtype=tf.float32)
            outputs2, last_states2 = tf.nn.dynamic_rnn(cell=self.cell2, dtype=tf.float32,
                                                       # sequence_length=[self.config["scope"]] * self.batch_p,
                                                       inputs=outputs1)
        w1 = tf.Variable(tf.random_normal([128, 1]), name="Wy")
        b1 = tf.Variable(tf.random_normal([1]), name="by")
        outputlink = outputs2[:, -1, :]
        # outputlink = tf.nn.dropout(outputlink, 0.8)
        y = tf.matmul(outputlink, w1) + b1
        # 损失返回值
        y_loss = tf.reduce_mean(tf.square(y - self.target_y), name="y_loss")

        # 部分2，预测操作
        w2 = tf.Variable(tf.random_normal([128, self.config["outspace"]]), name="Ws")
        b2 = tf.Variable(tf.random_normal([self.config["outspace"]]), name="bs")
        # 空间生成
        outspace = tf.add(tf.matmul(outputlink, w2), b2, name="outspace")
        # 空间的选择
        self.index_space = tf.argmax(outspace, axis=-1, output_type=dtypes.int32, name="outspace_id")
        # 空间的选择 损失函数
        space_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outspace, labels=self.space_chice), name="space_loss")

        # 空间对应的奖励 如果对
        self.god_posi = tf.one_hot(indices=self.space_chice, depth=self.config["outspace"], axis=1)
        self.pred_posi = tf.one_hot(indices=self.index_space, depth=self.config["outspace"], axis=1)
        self.space_if_benefit = tf.stack([1 + y[:, 0], y[:, 0] / y[:, 0], 1 - y[:, 0]], axis=-1,
                                         name="space_if_benefit")
        self.god_benefit_list = tf.reduce_sum(self.space_if_benefit * self.god_posi, axis=-1)
        # 猜错的获取 实际盈利值的负数
        norightorzero = tf.logical_and(
            tf.not_equal(self.index_space, self.space_chice), tf.not_equal(self.index_space, 1))
        fail_bene = tf.reduce_sum(tf.boolean_mask(self.god_benefit_list, norightorzero))
        # 猜对的获取 实际盈利值
        yesright = tf.equal(self.index_space, self.space_chice)
        win_bene = tf.reduce_sum(tf.boolean_mask(self.god_benefit_list, yesright))
        # 总盈利：奖励 - 损失值
        pred_bene = tf.subtract(win_bene, fail_bene, name="pred_bene")
        # 假如全盈利：奖励 + 损失值
        god_benefit = tf.reduce_sum(self.god_benefit_list)
        # 百分系数减慢对之前y的影响
        bene_rate = pred_bene / god_benefit * 100
        # self.learn_rate = tf.Variable(self.learn_rate_p, name="lr", trainable=False)
        # self.update_lr = tf.assign(self.learn_rate, tf.multiply(self.lr_decay, self.learn_rate))
        self.train_list = [y_loss, space_loss, -bene_rate]
        self.valid_list = [y, self.index_space, pred_bene, god_benefit, bene_rate]
        self.pred_list = [y, self.index_space]

        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss', y_loss)  # 记录标量的变化
        tf.summary.scalar('pred_bene', pred_bene)  # 记录标量的变化
        tf.summary.scalar('space_loss', space_loss)  # 记录标量的变化
        return None

    def _cnn_dense_model(self):
        # 部分1，预测值
        dense1 = tf.layers.dense(inputs=self.input_p, units=128, activation=tf.nn.relu, name="layer_dense1")
        concat1 = tf.concat([self.input_p, dense1], 1, name='concat1')
        denseo1 = tf.nn.dropout(concat1, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense1', dense1)  # 记录标量的变化
        dense2 = tf.layers.dense(inputs=denseo1, units=512, activation=tf.nn.relu, name="layer_dense2")
        concat2 = tf.concat([self.input_p, dense1, dense2], 1, name='concat2')
        denseo2 = tf.nn.dropout(concat2, keep_prob=self.keep_prob_ph)
        dense3 = tf.layers.dense(inputs=denseo2, units=256, activation=tf.nn.relu, name="layer_dense3")
        denseo3 = tf.nn.dropout(dense3, keep_prob=self.keep_prob_ph)
        dense4 = tf.layers.dense(inputs=denseo3, units=128, activation=tf.nn.relu, name="layer_dense4")
        denseo4 = tf.nn.dropout(dense4, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense2', dense2)  # 记录标量的变化
        y_base = tf.layers.dense(inputs=denseo4, units=self.base_dim, activation=None, name="y_base")
        y_much = tf.layers.dense(inputs=denseo4, units=self.much_dim, activation=None, name="y_much")
        tf.summary.histogram('y_base', y_base)  # 记录标量的变化
        tf.summary.histogram('y_much', y_much)  # 记录标量的变化
        # 损失返回值
        y_loss_base = tf.reduce_mean(tf.square(y_base - self.target_base_y), name="y_loss_base")
        y_loss_much = tf.reduce_mean(tf.square(y_much - self.target_much_y), name="y_loss_much")
        # 猜错的获取 实际盈利值的负数
        # self.learn_rate = tf.Variable(self.learn_rate_p, name="lr", trainable=False)
        # self.update_lr = tf.assign(self.learn_rate, tf.multiply(self.lr_decay, self.learn_rate))
        self.train_list = [y_loss_base, y_loss_much]
        self.valid_list = [y_loss_base, y_loss_much]
        self.pred_list = [y_base, y_much]
        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss_base', y_loss_base)  # 记录标量的变化
        tf.summary.scalar('y_loss_much', y_loss_much)  # 记录标量的变化

    def _cnn_dense_more_model(self):
        # 部分1，预测值
        dense1 = tf.layers.dense(inputs=self.input_p, units=128, activation=tf.nn.relu, name="layer_dense1")
        concat1 = tf.concat([self.input_p, dense1], 1, name='concat1')
        denseo1 = tf.nn.dropout(concat1, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense1', dense1)  # 记录标量的变化
        dense2 = tf.layers.dense(inputs=denseo1, units=512, activation=tf.nn.relu, name="layer_dense2")
        concat2 = tf.concat([self.input_p, dense1, dense2], 1, name='concat2')
        denseo2 = tf.nn.dropout(concat2, keep_prob=self.keep_prob_ph)
        dense3 = tf.layers.dense(inputs=denseo2, units=256, activation=tf.nn.relu, name="layer_dense3")
        concat3 = tf.concat([self.input_p, dense1, dense2, dense3], 1, name='concat3')
        denseo3 = tf.nn.dropout(concat3, keep_prob=self.keep_prob_ph)
        dense4 = tf.layers.dense(inputs=denseo3, units=128, activation=tf.nn.relu, name="layer_dense4")
        denseo4 = tf.nn.dropout(dense4, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense2', dense2)  # 记录标量的变化
        y_reta = tf.layers.dense(inputs=denseo4, units=self.out_dim, activation=None, name="y_reta")
        y_reth = tf.layers.dense(inputs=denseo4, units=self.out_dim, activation=None, name="y_reth")
        y_retl = tf.layers.dense(inputs=denseo4, units=self.out_dim, activation=None, name="y_retl")
        y_stdup = tf.layers.dense(inputs=denseo4, units=self.out_dim, activation=None, name="y_stdup")
        y_stddw = tf.layers.dense(inputs=denseo4, units=self.out_dim, activation=None, name="y_stddw")
        y_drawup = tf.layers.dense(inputs=denseo4, units=self.out_dim, activation=None, name="y_drawup")
        y_drawdw = tf.layers.dense(inputs=denseo4, units=self.out_dim, activation=None, name="y_drawdw")
        tf.summary.histogram('y_reta', y_reta)  # 记录标量的变化
        tf.summary.histogram('y_reth', y_reth)  # 记录标量的变化
        tf.summary.histogram('y_retl', y_retl)  # 记录标量的变化
        tf.summary.histogram('y_stdup', y_stdup)  # 记录标量的变化
        tf.summary.histogram('y_stddw', y_stddw)  # 记录标量的变化
        tf.summary.histogram('y_drawup', y_drawup)  # 记录标量的变化
        tf.summary.histogram('y_drawdw', y_drawdw)  # 记录标量的变化
        # 损失返回值
        y_loss_reta = tf.reduce_mean(tf.square(y_reta - self.reta), name="y_loss_reta")
        y_loss_reth = tf.reduce_mean(tf.square(y_reth - self.reth), name="y_loss_reth")
        y_loss_retl = tf.reduce_mean(tf.square(y_retl - self.retl), name="y_loss_retl")
        y_loss_stdup = tf.reduce_mean(tf.square(y_stdup - self.stdup), name="y_loss_stdup")
        y_loss_stddw = tf.reduce_mean(tf.square(y_stddw - self.stddw), name="y_loss_stddw")
        y_loss_drawup = tf.reduce_mean(tf.square(y_drawup - self.drawup), name="y_loss_drawup")
        y_loss_drawdw = tf.reduce_mean(tf.square(y_drawdw - self.drawdw), name="y_loss_drawdw")
        # 猜错的获取 实际盈利值的负数
        # self.learn_rate = tf.Variable(self.learn_rate_p, name="lr", trainable=False)
        # self.update_lr = tf.assign(self.learn_rate, tf.multiply(self.lr_decay, self.learn_rate))
        self.train_list = [y_loss_reta, y_loss_reth, y_loss_retl, y_loss_stdup, y_loss_stddw, y_loss_drawup,
                           y_loss_drawdw]
        self.valid_list = [y_loss_reta, y_loss_reth, y_loss_retl, y_loss_stdup, y_loss_stddw, y_loss_drawup,
                           y_loss_drawdw]
        self.pred_list = [y_reta, y_reth, y_retl, y_stdup, y_stddw, y_drawup, y_drawdw]
        # 打印信息
        tf.summary.scalar('y_loss_reta', y_loss_reta)  # 记录标量的变化
        tf.summary.scalar('y_loss_reth', y_loss_reth)  # 记录标量的变化
        tf.summary.scalar('y_loss_retl', y_loss_retl)  # 记录标量的变化
        tf.summary.scalar('y_loss_stdup', y_loss_stdup)  # 记录标量的变化
        tf.summary.scalar('y_loss_stddw', y_loss_stddw)  # 记录标量的变化
        tf.summary.scalar('y_loss_drawup', y_loss_drawup)  # 记录标量的变化
        tf.summary.scalar('y_loss_drawdw', y_loss_drawdw)  # 记录标量的变化
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化

    def batch_train(self, inputs_t, reta_t, reth_t, retl_t, stdup_t, stddw_t, drawup_t, drawdw_t,
                    inputs_v, reta_v, reth_v, retl_v, stdup_v, stddw_v, drawup_v, drawdw_v,
                    batch_size=8, num_epochs=1, retrain=True):
        # 设置
        dataiter = batch_iter_list([inputs_t, reta_t, reth_t, retl_t, stdup_t, stddw_t, drawup_t, drawdw_t],
                                   batch_size, num_epochs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if self.config["retrain"] == 1:
                model_dir = os.path.join(model_path, "modelevery_%s" % self.config["tailname"])
                latest_ckpt = tf.train.latest_checkpoint(model_dir)
                if os.path.isfile("{}.index".format(latest_ckpt)):
                    self.saver.restore(sess, latest_ckpt)
                    print("retraining {}".format(latest_ckpt))
                else:
                    sess.run(tf.global_variables_initializer())
                    print("no old model, training new----")
            writer = tf.summary.FileWriter(os.path.join(log_path, "logsevery_%s" % self.config["tailname"]), sess.graph)
            global_n = 0
            stop_n = 0
            startt = time.time()
            pre_t_base_loss = pre_t_much_loss = pre_v_much_loss = pre_v_base_loss = 100000
            for epoch in range(num_epochs):
                starte = time.time()
                losslist = [0]
                for batch_num in range(inputs_t.shape[0] // batch_size + 1):
                    # 获取数据
                    r_inputs_t, r_reta_t, r_reth_t, r_retl_t, r_stdup_t, r_stddw_t, r_drawup_t, r_drawdw_t = next(dataiter)
                    feed_dict_t = {
                        self.input_p: r_inputs_t,
                        self.reta: r_reta_t,
                        self.reth: r_reth_t,
                        self.retl: r_retl_t,
                        self.stdup: r_stdup_t,
                        self.stddw: r_stddw_t,
                        self.drawup: r_drawup_t,
                        self.drawdw: r_drawdw_t,
                        self.learn_rate_p: self.config["learn_rate"],
                        self.lr_decay: 1,
                    }
                    # 更新学习率
                    # tmplr = sess.run(self.update_lr)
                    for _ in range(self.config["single_num"]):
                        # 更新速度
                        sess.run(self.train_op, feed_dict_t)
                        global_n += 1
                    losslist_t = sess.run(self.train_list, feed_dict_t)
                    result = sess.run(self.merged, feed_dict_t)
                    if batch_num % 20 == 0:
                        writer.add_summary(result, global_n)
                        self.saver.save(sess, os.path.join(model_path, 'model_%s' % self.config["tailname"],
                                                           self.config["modelfile"]), global_step=global_n)
                        print("epocht {}, step {}, time: {} s, loss_reta {}, loss_reth {}, loss_retl {}, "
                              "loss_stdup {}, loss_stddw {}, loss_drawup {}, loss_drawdw {}".format(
                            epoch, global_n, time.time() - starte, *losslist_t))
                # valid part
                feed_dict_v = {
                    self.input_p: inputs_v,
                    self.reta: reta_v,
                    self.reth: reth_v,
                    self.retl: retl_v,
                    self.stdup: stdup_v,
                    self.stddw: stddw_v,
                    self.drawup: drawup_v,
                    self.drawdw: drawdw_v,
                    self.lr_decay: 1,
                }
                losslist_v = sess.run(self.valid_list, feed_dict_v)
                if losslist_t[0] < pre_t_base_loss and losslist_v[0] < pre_v_base_loss:
                    stop_n += 1
                    if stop_n > self.config["early_stop"]:
                        break
                else:
                    stop_n = 0
                print("epochv {}, step {}, stop_n {}, time: {} s, loss_reta_v {}, loss_reth_v {}, loss_retl_v {}, "
                      "loss_stdup_v {}, loss_stddw_v {}, loss_drawup_v {}, loss_drawdw_v {}".format(
                    epoch, global_n, stop_n, time.time() - starte, *losslist_v))
                pre_t_base_loss = losslist_t[0]
                pre_t_much_loss = losslist_t[1]
                pre_v_base_loss = losslist_v[0]
                pre_v_much_loss = losslist_v[1]
            writer.close()
            print("total time: %s s" % (time.time() - startt))
        # 结束
        print("train finished!")
        return None

    def predict(self, inputs):
        # self.base_dim, self.much_dim = base_dim, much_dim
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model_dir = os.path.join(model_path, "modelevery_%s" % self.config["tailname"])
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, latest_ckpt)
            feed_dict = {
                self.input_p: inputs,
            }
            teslist = sess.run(self.pred_list, feed_dict)
            return teslist
