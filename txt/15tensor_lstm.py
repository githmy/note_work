import codecs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class LSTM_my():
    def __init__(self, config):
        # timeStep = 25
        # hiddenUnitSize = 10  # 隐藏层神经元数量
        # batchSize = 60  # 每一批次训练多少个样例
        # inputSize = 1  # 输入维度
        # outputSize = 1  # 输出维度
        # lr = 0.0006  # 学习率
        self.timeStep = config["timeStep"]
        self.hiddenUnitSize = config["hiddenUnitSize"]
        self.batchSize = config["batchSize"]
        self.inputSize = config["inputSize"]
        self.outputSize = config["outputSize"]
        self.lr_start = config["lr_start"]
        self.lr_min = config["lr_min"]
        self.lr_decay = config["lr_decay"]
        self.lr_num = config["lr_num"]

        self.X = tf.placeholder(tf.float32, [None, self.timeStep, self.inputSize])
        self.Y = tf.placeholder(tf.float32, [None, self.timeStep, self.inputSize])
        self.tflr = tf.placeholder(tf.float32, [])
        self.tflrdecay = tf.placeholder(tf.float32, [])
        self.weights = {
            'in': tf.Variable(tf.random_normal([self.inputSize, self.hiddenUnitSize])),
            'out': tf.Variable(tf.random_normal([self.hiddenUnitSize, 1]))
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.hiddenUnitSize, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

    # lstm算法定义
    def lstm(self, batchSize=None):
        if batchSize is None:
            batchSize = self.batchSize
        weightIn = self.weights['in']
        biasesIn = self.biases['in']
        input = tf.reshape(self.X, [-1, self.inputSize])
        inputRnn = tf.matmul(input, weightIn) + biasesIn
        inputRnn = tf.reshape(inputRnn, [-1, self.timeStep, self.hiddenUnitSize])  # 将tensor转成3维，作为lstm cell的输入
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenUnitSize)
        initState = cell.zero_state(batchSize, dtype=tf.float32)
        # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output_rnn, final_states = tf.nn.bidirectional_dynamic_rnn(cell, inputRnn, initial_state=initState,
                                                                   dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, self.hiddenUnitSize])  # 作为输出层的输入
        w_out = self.weights['out']
        b_out = self.biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states

    # 训练模型
    def trainLstm(self, epochs, train_x, train_y):
        pred, _ = self.lstm()
        # 定义损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1], name="predicts") - tf.reshape(self.Y, [-1])))
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
                        print("保存模型：", saver.save(sess, 'stock.model'))
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
    model = LSTM_my(config=config)
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
