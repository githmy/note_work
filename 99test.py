import codecs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Lambda


def kera_structure(config):
    # 1. 参数传递
    inputSize = config["inputSize"]
    inputSize_ex = config["inputSize_ex"]
    lr_start = config["lr_start"]
    num1 = inputSize // 4
    num2 = num1 // 4
    # 2. 结构定义
    main_input = Input((inputSize,), dtype='float32', name='main_input')
    aux_input = Input((inputSize_ex,), dtype='float32', name='aux_input')

    def var_trans(inputs):
        a, b = inputs
        return a + b

    vartt = K.random_uniform_variable(shape=(1, inputSize_ex), low=-1, high=1, dtype="float32")
    print(type(aux_input))
    print(type(vartt))
    x_ex_in = Lambda(var_trans, name='varlay')([aux_input, vartt])
    x_in = keras.layers.concatenate([main_input, x_ex_in])
    x_in = Dense(num1, activation="relu", name='output')(x_in)
    x_in = keras.layers.concatenate([x_in, -x_in])
    x_in = Dense(num2, activation="relu", name='output')(x_in)
    x_in = keras.layers.concatenate([x_in, -x_in])
    main_output = Dense(1, activation=None, name='output')(x_in)
    # 3. 输入输出定义
    model = Model(inputs=[main_input, aux_input], outputs=[main_output])
    model.summary()
    # 4. 编译
    adam = Adam(lr=lr_start, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
    model.compile(optimizer=adam,
                  loss={'main_output': 'mse'},
                  loss_weights={'main_output': 1},
                  metrics={'main_output': 'accuracy'})
    return model


def keras_train(model, traininput, traintarget, validinput, validtarget, paras):
    dummydata = np.zeros(paras["batchsize"], paras["inputSize_ex"])
    model.fit(x={'main_input': traininput, 'aux_input': dummydata},
              y={'main_output': traintarget},
              batch_size=paras["batchsize"], epochs=paras["epochs"], verbose=1)
    score = model.evaluate(x={'main_input': validinput},
                           y={'main_output': validtarget},
                           batch_size=paras["batchsize"], verbose=1)
    return model


def keras_predict(model, inputdata):
    prediction = model.predict(inputdata)
    print(prediction)
    print(score)


class Card_my(object):
    def __init__(self, config):
        # timeStep = 25
        # hiddenUnitSize = 10  # 隐藏层神经元数量
        # batchSize = 60  # 每一批次训练多少个样例
        # inputSize = 1  # 输入维度
        # outputSize = 1  # 输出维度
        # lr = 0.0006  # 学习率
        # self.timeStep = config["timeStep"]
        self.batchSize = config["batchSize"]
        self.inputSize = config["inputSize"]
        self.inputSize_ex = config["inputSize_ex"]
        self.outputSize = config["outputSize"]
        self.lr_start = config["lr_start"]
        self.lr_min = config["lr_min"]
        self.lr_decay = config["lr_decay"]
        self.lr_num = config["lr_num"]
        self.modelname = config["modelname"]

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
        num1 = self.inputSize // 4
        x = tf.keras.layers.Dense(num1, activation='relu')(self.X)
        num1 = num1 // 4
        print(x)
        x = tf.keras.layers.Dense(num1, activation='relu')(x)
        num1 = num1 // 4
        print(x)
        x = tf.keras.layers.Dense(num1, activation='relu')(x)
        num1 = num1 // 4
        print(x)
        self.y = tf.keras.layers.Dense(1, activation=None)(x)
        print(self.y)

    # 训练模型
    def train_net(self, epochs, train_x, train_y):
        # 定义损失函数
        loss = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.Y)))
        # 定义训练模型
        tmplr = tf.Variable(0., name="lr")
        tf.summary.scalar('lr', tmplr)  # 记录标量的变化
        uplr = tf.assign(tmplr, tf.multiply(self.tflrdecay, tmplr))
        train_op = tf.train.AdamOptimizer(tmplr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            # 重复训练100次，训练是一个耗时的过程
            for i in range(epochs):
                step = 0
                start = 0
                end = start + self.batchSize
                while end < len(train_x):
                    # 更新速度
                    if step % self.lr_num == 0 and tmplr > self.lr_min:
                        tmplr = sess.run([uplr], feed_dict={
                            self.tflr: self.lr_start,
                            self.tflrdecay: self.lr_decay,
                        })
                    # 训练
                    _, _, loss_ = sess.run([merged, train_op, loss],
                                           feed_dict={
                                               self.X: train_x[start:end],
                                               self.Y: train_y[start:end],
                                           })
                    start += self.batchSize
                    end = start + self.batchSize
                    # 每10步保存一次参数
                    if step % 10 == 0:
                        print(i, step, loss_)
                        print("保存模型：", saver.save(sess, self.modelname))
                    step += 1

    def trainLstm2(self, epochs, train_x, train_y):
        pred, _ = self.lstm()
        # 定义损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(self.Y, [-1])))
        # 定义训练模型
        tmplr = tf.Variable(0., name="lr")
        uplr = tf.assign(tmplr, tf.multiply(self.tflrdecay, tmplr))
        train_op = tf.train.AdamOptimizer(tmplr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tmplr = self.lr_start
            # 重复训练100次，训练是一个耗时的过程
            for i in range(epochs):
                step = 0
                start = 0
                end = start + self.batchSize
                while end < len(train_x):
                    # 更新速度
                    if step % self.lr_num == 0 and tmplr > self.lr_min:
                        tmplr = sess.run([uplr], feed_dict={
                            self.tflr: tmplr,
                            self.tflrdecay: self.lr_decay,
                        })
                    # 训练
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={
                                            self.X: train_x[start:end],
                                            self.Y: train_y[start:end],
                                        })
                    start += self.batchSize
                    end = start + self.batchSize
                    # 每10步保存一次参数
                    if step % 10 == 0:
                        print(i, step, loss_)
                        print("保存模型：", saver.save(sess, 'stock.model'))
                    step += 1

    def prediction(self, train_x):
        pred, _ = self.lstm(1)  # 预测时只输入[1,time_step,inputSize]的测试数据
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # 参数恢复
            module_file = tf.train.latest_checkpoint('./')
            saver.restore(sess, module_file)
            # 取训练集最后一行为测试样本. shape=[1,time_step,inputSize]
            prev_seq = train_x[-1]
            predict = []
            # 得到之后100个预测结果
            for i in range(100):
                next_seq = sess.run(pred, feed_dict={self.X: [prev_seq]})
                predict.append(next_seq[-1])
                # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))


if __name__ == "__main__":
    config = {
        "timeStep": 25,
        "hiddenUnitSize": 10,
        "batchSize": 60,
        "inputSize": 1,
        "outputSize": 1,
        "lr_start": 0.01,
        "lr_min": 0.000001,
        "lr_decay": 0.998,
        "lr_num": 1000,
    }
    model = Card_my(config=config)
    # model.loadData()
    # # 构建训练数据
    # model.buildTrainDataSet()
    # 模型训练
    epochs = 100
    train_x, train_y, test_x = [], [], []
    model.trainLstm(epochs, train_x, train_y)
    model.trainLstm2(epochs, train_x, train_y)

    # 预测－预测前需要先完成模型训练
    model.prediction(test_x)
