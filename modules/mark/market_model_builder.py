from model_builder import AbstractModeltensor
from tensorflow.python.framework import dtypes
import tensorflow as tf
import time
import os
import numpy as np


class RNN(AbstractModeltensor):
    def __init__(self, config=None):
        super(RNN, self).__init__(config)
        self.modeldic = {
            "full": self._fullmodel,  # 原始结构
            "one": self._one,  # 原始改单层结构
            "one_y": self._one_y,  # 原始改单层结构
            "one_space": self._one_space,  # 原始改单层结构
            "one_attent": self._one_attent,  # 原始改单层结构
            "one_attent60": self._one_attent60,  # 原始改单层结构
        }
        self.ydim = 1
        self.attention_size = 10
        self.keep_prob_ph = config["dropout"]
        with tf.name_scope('Inputs'):
            self.input_p = tf.placeholder(tf.float32, [None, self.config["scope"], self.config["inputdim"]])
            # self.target_p = tf.placeholder(dtype=tf.float32, shape=[None, self.config["inputdim"]])
            self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, self.ydim])
            self.space_chice = tf.placeholder(dtype=tf.int32, shape=[None])
            self.batch_p = tf.placeholder(dtype=tf.int32, shape=[])
            self.learn_rate_p = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            self.lr_decay = tf.placeholder(dtype=tf.float32, shape=[])
            self.normal = tf.placeholder(dtype=tf.float32, shape=[])

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

    def batch_train(self, inputs, targets, space_chice, global_n):
        # 设置
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter("logs_%s" % self.config["tailname"], sess.graph)
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                self.input_p: inputs,
                self.target_y: targets,
                # self.target_p: targets_p,
                self.space_chice: space_chice,
                self.batch_p: inputs.shape[0],
                self.learn_rate_p: self.config["learn_rate"],
                self.lr_decay: 1,
                self.normal: self.config["normal"],
            }
            startt = time.time()
            # 更新学习率
            # tmplr = sess.run(self.update_lr)
            for _ in range(self.config["single_num"]):
                # 更新速度
                sess.run(self.train_op, feed_dict)
                global_n += 1
            losslist = sess.run(self.train_list, feed_dict)
            result = sess.run(self.merged, feed_dict)
            writer.add_summary(result, global_n)
            self.saver.save(sess, os.path.join('model_%s' % self.config["tailname"], self.config["modelfile"]),
                            global_step=global_n)
            writer.close()
            # print(sess.run([self.space_if_benefit, self.god_benefit_list], feed_dict))
            print("step %s, time: %s s, y_loss %s, space_cost %s, -bene_rate %s" % (
                global_n, time.time() - startt, losslist[0], losslist[1], losslist[2]))
        # 返回全局步数
        return global_n

    def valid_test(self, inputs, targets, space_chice):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                self.input_p: inputs,
                self.target_y: targets,
                self.space_chice: space_chice,
                self.batch_p: inputs.shape[0],
            }
            valist = sess.run(self.valid_list, feed_dict)
            return valist

    def predict(self, inputs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                self.input_p: inputs,
                self.batch_p: inputs.shape[0],
            }
            teslist = sess.run(self.pred_list, feed_dict)
            return teslist
