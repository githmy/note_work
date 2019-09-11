# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc
import datetime
import os
import time
import tensorflow as tf
from utils.log_tool import model_path, log_path, data_path, conf_path


# 构建模型
class TrendNN(object):
    def __init__(self, model_type, model_name, model_json, curvesobj, lenlist):
        if model_type is None:
            self.model_type = "default"
        else:
            self.model_type = model_type
        if model_name is None:
            self.model_name = time.strftime("%Y%m%d%H%M%S", time.localtime())
        else:
            self.model_name = model_name
        self.model_instance_name = "{}-{}".format(self.model_type, self.model_name)
        self._model_json = model_json
        # 参数转内部
        self.num_epochs = self._model_json["num_epochs"]
        self.learning_rate = self._model_json["learning_rate"]
        self.evaluate_every = self._model_json["evaluate_every"]
        self.early_stop = self._model_json["early_stop"]
        self.save_step = self._model_json["save_step"]
        self.default_g = None
        # 路径拼接
        self.model_dir = os.path.join(model_path, self.model_instance_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.log_dir = os.path.join(log_path, self.model_instance_name)
        self.data_dir = os.path.join(data_path, self.model_instance_name)
        self.conf_dir = os.path.join(conf_path, self.model_instance_name)
        # 转入数据
        self.curvesobj = curvesobj
        self.lenlist = lenlist

    def build(self):
        self.curveshape = self.curvesobj.shape

        def get_point_loss(xs, xl, a, b, c, m):
            xst = tf.transpose(xs[:, :, 0]) + m
            xst = tf.transpose(xst)
            funcmani = a * tf.square(xst) + b * xst + c - xs[:, :, 1]
            loss_all = tf.reduce_sum(tf.square(funcmani * xl))
            return loss_all

        # 1. 输入层
        with tf.name_scope('input'):
            self.input_xs = tf.placeholder(tf.float32, [self.curveshape[0], self.curveshape[1], self.curveshape[2]],
                                           name='input_xs')
            self.input_xl = tf.placeholder(tf.float32, [self.curveshape[0], self.curveshape[1]], name='input_xl')
            # 2. 输出
        with tf.name_scope("process"):
            self.a = tf.Variable(tf.constant(-5.0), name='a')
            self.b = tf.Variable(tf.constant(3.0), name='b')
            self.c = tf.Variable(tf.constant(2.0), name='c')
            self.m = tf.Variable(tf.random_normal([self.curveshape[0]], stddev=0.35, name='m'))
            self.l = get_point_loss(self.input_xs, self.input_xl, self.a, self.b, self.c, self.m)

        tf.summary.scalar('a', self.a)
        tf.summary.scalar('b', self.b)
        tf.summary.scalar('c', self.c)
        tf.summary.scalar('loss', self.l)
        tf.summary.histogram('move', self.m)

    def fit(self):
        model_name = os.path.join(self.model_dir, "{}.ckpt".format(self.model_instance_name))
        best_loss = 1e9
        best_counter = 0
        # 9. 辅助参数
        with tf.name_scope("assistant"):
            train1 = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.l)
            merged = tf.summary.merge_all()
            self.train_op = [merged, train1]
            self.res_list = [self.a, self.b, self.c, self.m, self.l]
        if 1:
            res_a, res_b, res_c, res_m = None, None, None, None
            session_conf = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True),
                allow_soft_placement=True,  # 如果指定的设备不存在，允许tf自动分配设备
                log_device_placement=False)  # 不打印设备分配日志
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.early_stop)
            with tf.Session(config=session_conf) as sess:
                # 2. 用于统计全局的step
                print("start".center(20, " ").center(80, "*"))
                train_writer = tf.summary.FileWriter(self.log_dir, sess.graph)  # 保存位置
                sess.run(tf.global_variables_initializer())
                counter = 0  # 用于记录当前的batch数
                for step in range(self.num_epochs):
                    counter += 1
                    # 训练
                    feed_dict = {
                        self.input_xs: self.curvesobj,
                        self.input_xl: self.lenlist,
                    }
                    summary1, _ = sess.run(self.train_op, feed_dict)
                    if step % self.save_step == 0:
                        train_writer.add_summary(summary1, step)
                        print("保存模型：", saver.save(sess, model_name, step))
                    if step % self.evaluate_every == 0:
                        res_a, res_b, res_c, res_m, res_l = sess.run(self.res_list, feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, a {:g}, b {:g}, c {:g}".format(time_str, step, res_a, res_b, res_c))
                        print(res_m)
                        # 每10步保存一次参数
                        if best_loss > res_l:
                            best_loss = res_l
                            best_counter = 0
                        else:
                            best_counter += 1
                            if self.early_stop < best_counter:
                                train_writer.add_summary(summary1, step)
                                print("保存模型：", saver.save(sess, model_name, step))
                                break
                        print("best_loss: %s, best_counter: %s" % (best_loss, best_counter))
                print("training finished!")
                train_writer.close()
                return res_a, res_b, res_c, res_m

    def load_mode(self, modelname):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            latest_ckpt = tf.train.latest_checkpoint(self.model_dir)
            if latest_ckpt:
                """Load model from a checkpoint."""
                try:
                    # 加载模型
                    saver.restore(sess, latest_ckpt)
                except tf.errors.NotFoundError as e:
                    print("Can't load checkpoint")
                    # 预读自定义权重


def main():
    pass


if __name__ == '__main__':
    main()
