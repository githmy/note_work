import codecs
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Lambda


# def test():
#     print(sum(range(1, 101)))
#
#
# if __name__ == '__main__':
#     test()
#     exit(0)
"如图，在△ABC中，D，E分别在边AC与AB上，DE∥BC，BD、CE相交于点O，已知∠AOB=25°42'，则∠AOB的余角为"

class Card_my(object):
    def __init__(self, config):
        self.batchSize = config["batchSize"]
        self.inputSize = config["inputSize"]
        self.inputSize_ex = config["inputSize_ex"]
        self.outputSize = config["outputSize"]
        self.lr_start = config["lr_start"]
        self.lr_min = config["lr_min"]
        self.lr_decay = config["lr_decay"]
        self.lr_num = config["lr_num"]
        self.modelname = config["modelname"]
        self.nocodepath = config["nocodepath"]
        self.earlystop = config["earlystop"]
        self.epochs = config["epochs"]

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, self.inputSize])
            self.X_ex = tf.placeholder(tf.float32, [None, self.inputSize_ex])
            self.tflr = tf.placeholder(tf.float32, [])
            self.tflrdecay = tf.placeholder(tf.float32, [])
        with tf.name_scope('output'):
            self.Y = tf.placeholder(tf.float32, [None, 1])
        # 模型加载
        self.net_structure()

    # lstm算法定义
    def net_structure(self):
        # with tf.name_scope('layer1'):
        #     Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        #     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        var_ex = tf.Variable(tf.random_normal([1, self.outputSize]), name='b')
        in_new = tf.concat([self.X, self.X_ex + var_ex], 1)
        num1 = self.inputSize // 4
        x = tf.keras.layers.Dense(num1, activation='relu')(in_new)
        x = tf.concat([x, -x], 1)
        num1 = num1 // 4
        print(x)
        x = tf.keras.layers.Dense(num1, activation='relu')(x)
        x = tf.concat([x, -x], 1)
        num1 = num1 // 4
        print(x)
        x = tf.keras.layers.Dense(num1, activation='relu')(x)
        x = tf.concat([x, -x], 1)
        num1 = num1 // 4
        print(x)
        self.y = tf.keras.layers.Dense(1, activation=None)(x)
        print(self.y)

    # 训练模型
    def train_net(self, datamap):
        # 1. 变量预设
        counter = 0
        train_length = len(datamap["train_x"])
        valid_length = len(datamap["valid_x"])
        # 2. 定义损失函数
        loss = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.Y)))
        # 3. 定义训练模型
        tmplr = tf.Variable(0., name="lr")
        tf.summary.scalar('lr', tmplr)  # 记录标量的变化
        uplr = tf.assign(tmplr, tf.multiply(self.tflrdecay, tmplr))
        train_op = tf.train.AdamOptimizer(tmplr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        # 4. 训练
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(self.nocodepath, "logs", "train"), sess.graph)
            sess.run(tf.global_variables_initializer())
            # 4.1 重复训练100次
            for i in range(self.epochs):
                step = 0
                start = 0
                end = start + self.batchSize
                while start < train_length:
                    if end > train_length:
                        end = train_length
                    # 4.1.1 更新速度，和数据
                    if step % self.lr_num == 0 and tmplr > self.lr_min:
                        tmplr = sess.run([uplr], feed_dict={
                            self.tflr: self.lr_start,
                            self.tflrdecay: self.lr_decay,
                        })
                    # 4.1.2 训练
                    mer_log, _, loss_ = sess.run([merged, train_op, loss],
                                                 feed_dict={
                                                     self.X: datamap["train_x"][start:end],
                                                     self.X_ex: datamap["train_ex"],
                                                     self.Y: datamap["train_y"][start:end],
                                                 })
                    start += self.batchSize
                    end = start + self.batchSize
                    # 4.1.3 每100步保存一次参数，并做验证
                    if step % 100 == 0:
                        # 4.1.3.1 验证循环
                        step_v = 0
                        start_v = 0
                        end_v = start_v + self.batchSize
                        valid_arr = np.array()
                        validloss_store = 1e9
                        while start_v < valid_length:
                            if end_v > valid_length:
                                end_v = valid_length
                            loss_s = sess.run([loss],
                                              feed_dict={
                                                  self.X: datamap["valid_x"][start_v:end_v],
                                                  self.X_ex: datamap["train_ex"],
                                                  self.Y: datamap["valid_y"][start_v:end_v],
                                              })
                            valid_arr = np.vstack((valid_arr, loss_s))
                            start_v += self.batchSize
                            end_v = start_v + self.batchSize
                        val_loss = valid_arr.mean()
                        # 4.1.3.2 信息打印
                        print("epoch: %s, step: %s, train: %s, valid: %s " % (i, step, loss_, val_loss))
                        train_writer.add_summary(mer_log, step + i * train_length)
                        print("保存模型：",
                              saver.save(sess, os.path.join(self.nocodepath, "model", self.modelname + "_" + val_loss)))
                        if val_loss > validloss_store:
                            counter += 1
                            if counter > self.earlystop:
                                train_writer.close()
                                print("no more promotion.")
                                return 0
                        else:
                            validloss_store = val_loss

                    step += 1
            train_writer.close()

    def prediction(self, datamap):
        # 1. 变量预设
        test_length = len(datamap["test_x"])
        # 2. 定义损失函数
        loss = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.Y)))
        # 4. 训练
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            # 4.1 重复训练100次
            for i in range(self.epochs):
                start = 0
                end = start + self.batchSize
                test_arr = np.array()
                while start < test_length:
                    if end > test_length:
                        end = test_length
                    # 4.1.2 训练
                    out = sess.run(self.y, feed_dict={
                        self.X: datamap["test_x"][start:end],
                        self.X_ex: datamap["train_ex"],
                    })
                    test_arr = np.vstack((test_arr, out))
                    start += self.batchSize
                    end = start + self.batchSize
        return test_arr


if __name__ == "__main__":
    config = {
        "timeStep": 25,
        "epochs": 128,
        "earlystop": 3,
        "nocodepath": os.path.join(".."),
        "modelname": "card_model",
        "batchSize": 60,
        "inputSize": 1000,
        "inputSize_ex": 5,
        "outputSize": 1,
        "lr_start": 1e-5,
        "lr_min": 0.000001,
        "lr_decay": 0.998,
        "lr_num": 1000,
    }

    train_x, train_y, valid_x, valid_y, test_x = [], [], [], [], []
    train_ex = np.zeros(shape=(config["batchSize"], config["inputSize_ex"]))
    datamap = {
        "train_ex": train_ex,
        "train_x": train_x,
        "train_y": train_y,
        "valid_x": valid_x,
        "valid_y": valid_y,
        "test_x": test_x,
    }
    model = Card_my(config=config)
    # model.loadData()
    # # 构建训练数据
    # model.buildTrainDataSet()
    # 模型训练
    model.train_net(datamap)

    # 预测－预测前需要先完成模型训练
    model.prediction(datamap)
