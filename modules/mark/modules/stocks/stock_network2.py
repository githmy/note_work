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

    def getModel(self):
        model = self.buildModel()
        model_dir = os.path.join(model_path, "model_%s" % self.config["tailname"])
        if model_dir and os.path.isdir(model_dir):
            try:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                saver = tf.train.Saver()
                latest_ckpt = tf.train.latest_checkpoint(model_dir)
                saver.restore(sess, latest_ckpt)
            except Exception as e:
                print(e)
        return model

    # You need to override this method.
    def buildModel(self):
        raise NotImplementedError("You need to implement your own model.")


class CRNN(AbstractModeltensor):
    def __init__(self, ave_list, bband_list, config=None):
        super(CRNN, self).__init__(config)
        self.modeldic = {
            "cnn_dense": self._cnn_dense_model,  # 原始结构
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
        self.ret_dim, self.std_dim = len(bband_list) * len(bband_list), len(bband_list) * 6
        self.input_dim = len(ave_list) * (2 * len(ave_list) + 2)
        with tf.name_scope('Inputs'):
            self.input_p = tf.placeholder(tf.float32, [None, self.input_dim])
            self.learn_rate_p = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            self.lr_decay = tf.placeholder(dtype=tf.float32, shape=[])
        with tf.name_scope('Outputs'):
            self.target_ret_y = tf.placeholder(dtype=tf.float32, shape=[None, self.ret_dim])
            self.target_std_y = tf.placeholder(dtype=tf.float32, shape=[None, self.std_dim])

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
        dense1 = tf.layers.dense(inputs=self.input_p, units=1024, activation=tf.nn.relu, name="layer_dense1")
        denseo1 = tf.nn.dropout(dense1, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense1', dense1)  # 记录标量的变化
        dense2 = tf.layers.dense(inputs=denseo1, units=512, activation=tf.nn.relu, name="layer_dense2")
        denseo2 = tf.nn.dropout(dense2, keep_prob=self.keep_prob_ph)
        # tf.summary.histogram('layer_dense2', dense2)  # 记录标量的变化
        y_ret = tf.layers.dense(inputs=denseo2, units=self.ret_dim, activation=None, name="y_ret")
        y_std = tf.layers.dense(inputs=denseo2, units=self.std_dim, activation=None, name="y_std")
        tf.summary.histogram('y_ret', y_ret)  # 记录标量的变化
        tf.summary.histogram('y_std', y_std)  # 记录标量的变化
        # 损失返回值
        y_loss_ret = tf.reduce_mean(tf.square(y_ret - self.target_ret_y), name="y_loss_ret")
        y_loss_std = tf.reduce_mean(tf.square(y_std - self.target_std_y), name="y_loss_std")
        # 猜错的获取 实际盈利值的负数
        # self.learn_rate = tf.Variable(self.learn_rate_p, name="lr", trainable=False)
        # self.update_lr = tf.assign(self.learn_rate, tf.multiply(self.lr_decay, self.learn_rate))
        self.train_list = [y_loss_ret, y_loss_std]
        self.valid_list = [y_loss_ret, y_loss_std]
        self.pred_list = [y_ret, y_std]
        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss_ret', y_loss_ret)  # 记录标量的变化
        tf.summary.scalar('y_loss_std', y_loss_std)  # 记录标量的变化

    def _fullmodel(self):
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

    def _one_attent60(self):
        # 部分1，预测值
        input_p = self.input_p
        attention_size = self.attention_size
        hidden_size = self.config["inputdim"]
        time_lenth = self.config["scope"]
        with tf.name_scope('attention_scope_part'):
            # Trainable parameters
            w_omega = tf.Variable(
                tf.random_normal([time_lenth, hidden_size, attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([time_lenth, attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([time_lenth, attention_size], stddev=0.1))

            with tf.name_scope('v'):
                vu_list = []
                for i2 in range(time_lenth):
                    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                    #  the shape of `v` is (B,D)*(D,A)=(B,A), where A=attention_size
                    v = tf.tanh(tf.tensordot(input_p[:, i2, :], w_omega[i2, :, :], axes=1) + b_omega[i2, :])

                    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
                    vu = tf.tensordot(v, u_omega[i2, :], axes=1, name='vu')  # (B) shape
                    vu_list.append(vu)

                scalas = tf.nn.softmax(tf.stack(vu_list, axis=1), name='scalas')  # (B,T) shape
                # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
                hidden_out = tf.reduce_sum(input_p * tf.expand_dims(scalas, -1), 1)

        tf.summary.histogram('scalas', scalas)
        # hidden_out [B, att_hidden]
        # scalas [B, scope]
        drop = tf.nn.dropout(hidden_out, self.keep_prob_ph)

        # 部分2，预测操作
        with tf.name_scope('Full_space'):
            Ws = tf.Variable(tf.truncated_normal([hidden_size, self.config["outspace"]], stddev=0.02),
                             name="Ws")
            bs = tf.Variable(tf.constant(0., shape=[self.config["outspace"]]), name="bs")
            outspace = tf.nn.xw_plus_b(drop, Ws, bs, name="outspace")
            # ys_hat = tf.squeeze(ys_hat)
        with tf.name_scope('Full_y'):
            Wy = tf.Variable(tf.truncated_normal([hidden_size, self.ydim], stddev=0.02))
            by = tf.Variable(tf.constant(0., shape=[self.ydim]))
            tf.summary.histogram('Wy', Wy)
            y = tf.nn.xw_plus_b(drop, Wy, by)

        y_loss = tf.reduce_mean(tf.square(y - self.target_y) + self.config["normal"] * self.config["normal"],
                                name="y_loss")
        # 空间的选择
        self.index_space = tf.argmax(outspace, axis=-1, output_type=dtypes.int32, name="outspace_id")
        # 空间的选择 损失函数
        space_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outspace, labels=self.space_chice) + self.config[
                "normal"], name="space_loss")

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
        bene_rate = pred_bene / god_benefit
        self.train_list = [y_loss, space_loss, -bene_rate]
        self.valid_list = [y, self.index_space, pred_bene, god_benefit, bene_rate]
        self.pred_list = [y, self.index_space]

        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss', y_loss)  # 记录标量的变化
        tf.summary.scalar('pred_bene', pred_bene)  # 记录标量的变化
        tf.summary.scalar('space_loss', space_loss)  # 记录标量的变化

    def _one_attent(self):
        # 部分1，预测值
        input_p = self.input_p
        attention_size = self.attention_size
        hidden_size = self.config["inputdim"]

        with tf.name_scope('attention_part'):
            # Trainable parameters
            w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(input_p, w_omega, axes=1) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            scalas = tf.nn.softmax(vu, name='scalas')  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            hidden_out = tf.reduce_sum(input_p * tf.expand_dims(scalas, -1), 1)

        tf.summary.histogram('scalas', scalas)
        # hidden_out [B, att_hidden]
        # scalas [B, scope]
        drop = tf.nn.dropout(hidden_out, self.keep_prob_ph)

        # 部分2，预测操作
        with tf.name_scope('Full_space'):
            Ws = tf.Variable(tf.truncated_normal([hidden_size, self.config["outspace"]], stddev=0.02),
                             name="Ws")
            bs = tf.Variable(tf.constant(0., shape=[self.config["outspace"]]), name="bs")
            outspace = tf.nn.xw_plus_b(drop, Ws, bs, name="outspace")
            # ys_hat = tf.squeeze(ys_hat)
        with tf.name_scope('Full_y'):
            Wy = tf.Variable(tf.truncated_normal([hidden_size, self.ydim], stddev=0.02))
            by = tf.Variable(tf.constant(0., shape=[self.ydim]))
            tf.summary.histogram('Wy', Wy)
            y = tf.nn.xw_plus_b(drop, Wy, by)

        y_loss = tf.reduce_mean(tf.square(y - self.target_y), name="y_loss")
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
        bene_rate = pred_bene / god_benefit
        self.train_list = [y_loss, space_loss, -bene_rate]
        self.valid_list = [y, self.index_space, pred_bene, god_benefit, bene_rate]
        self.pred_list = [y, self.index_space]

        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss', y_loss)  # 记录标量的变化
        tf.summary.scalar('pred_bene', pred_bene)  # 记录标量的变化
        tf.summary.scalar('space_loss', space_loss)  # 记录标量的变化

    def _one_y(self):
        # 部分1，预测值
        with tf.variable_scope('lstm1', initializer=tf.random_normal_initializer()):
            self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
            state1 = self.cell1.zero_state(batch_size=self.batch_p, dtype=tf.float32)
            outputs1, last_states1 = tf.nn.dynamic_rnn(cell=self.cell1, dtype=tf.float32,
                                                       # sequence_length=[self.config["scope"]] * self.batch_p,
                                                       inputs=self.input_p)
        w1 = tf.Variable(tf.random_normal([64, 1]), name="Wy")
        b1 = tf.Variable(tf.random_normal([1]), name="by")
        outputlink = outputs1[:, -1, :]
        # outputlink = tf.nn.dropout(outputlink, 0.8)
        y = tf.matmul(outputlink, w1) + b1
        # 损失返回值
        y_loss = tf.reduce_mean(tf.square(y - self.target_y), name="y_loss")

        self.train_list = [y_loss, y_loss, y_loss]
        self.valid_list = [y, y, y, y, y]
        self.pred_list = [y, y]

        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss', y_loss)  # 记录标量的变化

    def _one_space(self):
        # 部分1，预测值
        with tf.variable_scope('lstm1', initializer=tf.random_normal_initializer()):
            self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
            state1 = self.cell1.zero_state(batch_size=self.batch_p, dtype=tf.float32)
            outputs1, last_states1 = tf.nn.dynamic_rnn(cell=self.cell1, dtype=tf.float32,
                                                       # sequence_length=[self.config["scope"]] * self.batch_p,
                                                       inputs=self.input_p)
        outputlink = outputs1[:, -1, :]
        # outputlink = tf.nn.dropout(outputlink, 0.8)

        # 部分2，预测操作
        w2 = tf.Variable(tf.random_normal([64, self.config["outspace"]]), name="Ws")
        b2 = tf.Variable(tf.random_normal([self.config["outspace"]]), name="bs")
        # 空间生成
        outspace = tf.add(tf.matmul(outputlink, w2), b2, name="outspace")
        # 空间的选择
        self.index_space = tf.argmax(outspace, axis=-1, output_type=dtypes.int32, name="outspace_id")
        # 空间的选择 损失函数
        space_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outspace, labels=self.space_chice), name="space_loss")

        self.train_list = [space_loss, space_loss, space_loss]
        self.valid_list = [self.index_space, self.index_space, self.index_space, self.index_space, self.index_space]
        self.pred_list = [self.index_space, self.index_space]

        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('space_loss', space_loss)  # 记录标量的变化

    def _one(self):
        # 部分1，预测值
        with tf.variable_scope('lstm1', initializer=tf.random_normal_initializer()):
            self.cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)
            state1 = self.cell1.zero_state(batch_size=self.batch_p, dtype=tf.float32)
            outputs1, last_states1 = tf.nn.dynamic_rnn(cell=self.cell1, dtype=tf.float32,
                                                       # sequence_length=[self.config["scope"]] * self.batch_p,
                                                       inputs=self.input_p)
        w1 = tf.Variable(tf.random_normal([64, 1]), name="Wy")
        b1 = tf.Variable(tf.random_normal([1]), name="by")
        outputlink = outputs1[:, -1, :]
        # outputlink = tf.nn.dropout(outputlink, 0.8)
        y = tf.matmul(outputlink, w1) + b1
        # 损失返回值
        y_loss = tf.reduce_mean(tf.square(y - self.target_y), name="y_loss")

        # 部分2，预测操作
        w2 = tf.Variable(tf.random_normal([64, self.config["outspace"]]), name="Ws")
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
        self.train_list = [y_loss, space_loss, -bene_rate]
        self.valid_list = [y, self.index_space, pred_bene, god_benefit, bene_rate]
        self.pred_list = [y, self.index_space]

        # 打印信息
        tf.summary.scalar('lr', self.learn_rate_p)  # 记录标量的变化
        tf.summary.scalar('y_loss', y_loss)  # 记录标量的变化
        tf.summary.scalar('pred_bene', pred_bene)  # 记录标量的变化
        tf.summary.scalar('space_loss', space_loss)  # 记录标量的变化

    def batch_train(self, inputs_t, targets_ret_t, targets_std_t, inputs_v, targets_ret_v, targets_std_v, batch_size=8,
                    num_epochs=1):
        # 设置
        dataiter = batch_iter_list([inputs_t, targets_ret_t, targets_std_t], batch_size, num_epochs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter(os.path.join(log_path, "logs_%s" % self.config["tailname"]), sess.graph)
            sess.run(tf.global_variables_initializer())
            global_n = 0
            stop_n = 0
            startt = time.time()
            pre_t_ret_loss = pre_t_std_loss = pre_v_std_loss = pre_v_ret_loss = 100000
            for epoch in range(num_epochs):
                starte = time.time()
                losslist = [0]
                for batch_num in range(inputs_t.shape[0] // batch_size + 1):
                    # 获取数据
                    inputs_x_t, inputs_yret_t, inputs_ystd_t = next(dataiter)
                    feed_dict_t = {
                        self.input_p: inputs_x_t,
                        self.target_ret_y: inputs_yret_t,
                        self.target_std_y: inputs_ystd_t,
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
                    writer.add_summary(result, global_n)
                    self.saver.save(sess, os.path.join(model_path, 'model_%s' % self.config["tailname"],
                                                       self.config["modelfile"]), global_step=global_n)
                    print("batch %s, step %s, time: %s s, y_loss_ret_t %s, y_loss_std_t %s" % (
                        batch_num, global_n, time.time() - starte, losslist_t[0], losslist_t[1]))
                # valid part
                feed_dict_v = {
                    self.input_p: inputs_v,
                    self.target_ret_y: targets_ret_v,
                    self.target_std_y: targets_std_v,
                    self.learn_rate_p: self.config["learn_rate"],
                    self.lr_decay: 1,
                }
                losslist_v = sess.run(self.valid_list, feed_dict_v)
                if losslist_t[0] < pre_t_ret_loss and losslist_v[0] < pre_v_ret_loss:
                    stop_n += 1
                    if stop_n > self.config["early_stop"]:
                        break
                print("epoch %s, step %s, stop_n %s, time: %s s, y_loss_ret_v %s, y_loss_std_v %s" % (
                    epoch, global_n, stop_n, time.time() - starte, losslist_v[0], losslist_v[1]))
                pre_t_ret_loss = losslist_t[0]
                pre_t_std_loss = losslist_t[1]
                pre_v_ret_loss = losslist_v[0]
                pre_v_std_loss = losslist_v[1]
            writer.close()
            print("total time: %s s" % (time.time() - startt))
        # 结束
        print("train finished!")
        return None

    def valid_test(self, inputs, targets, space_chice):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                self.input_p: inputs,
                self.target_y: targets,
                self.space_chice: space_chice,
            }
            valist = sess.run(self.valid_list, feed_dict)
            return valist

    def predict(self, inputs):
        # self.ret_dim, self.std_dim = ret_dim, std_dim
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                self.input_p: inputs,
            }
            teslist = sess.run(self.pred_list, feed_dict)
            return teslist
