# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf


def GPU_device():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config.gpu_options.allow_growth = True
    sess0 = tf.InteractiveSession(config=config)


# GPU操作
def gpu_setting():
    init = tf.global_variables_initializer()
    # 设置tensorflow对GPU的使用按需分配
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 2.启动图 (graph)
    sess = tf.Session(config=config)
    # sess = tf.InteractiveSession(config=config)
    sess.run(init)
    # #
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        pass


def csv_read():
    """
    
    :return: 
    """
    pass


def read_tfrecord(example_proto):
    """
    Decode TFRecords for Dataset.

    Args:
      example_proto: TensorFlow ExampleProto object. 

    Return:
      The op of features and labels
    """
    # 1. 加载数据文件
    features = {
        "features": tf.FixedLenFeature([FLAGS.feature_size], tf.float32),
        "label": tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["features"], parsed_features["label"]


def neurous_network(label_pd, batch_size=10):
    # 1. 获取数据
    xcol = [i1 for i1 in label_pd.columns if not i1.startswith("ylabel_")]
    ycol = [i1 for i1 in label_pd.columns if i1.startswith("ylabel_")]
    Xt = label_pd[xcol]
    Yt = label_pd[ycol]
    # 2. 学习数据
    M = label_pd.shape[0]
    Nf = len(xcol)  # train data feature dimension
    Nt = len(ycol)  # label feature dimension
    # w_data = np.mat([[1.0, 3.0]]).T
    # x_data = np.random.randn(M, N).astype(np.float32)
    x_data = np.array(Xt)
    # y_data = np.mat(x_data) * w_data + 10 + np.random.randn(M, 1) * 0.33
    y_data = np.array(Yt)

    # 如果存在旧图，重置。
    tf.reset_default_graph()
    # run model use session
    with tf.Session() as sess:
        with tf.name_scope("name_scope_test"):
            # define model graph and loss function
            # use tf tensor type var just like use np.array
            # X = tf.placeholder("float", [batch_size, Nf])
            with tf.name_scope("X"):
                X = tf.placeholder("float", [M, Nf], name='x_hold')
                tf.summary.histogram("name_scope_test" + '/X', X)
            # declare a graph node, but not init it immediately.
            # Y = tf.placeholder("float", [batch_size, len(ycol)])
            with tf.name_scope("Y"):
                Y = tf.placeholder("float", [M, len(ycol)], name='y_hold')
                tf.summary.histogram("name_scope_test" + '/Y', Y)
            with tf.name_scope("w"):
                w = tf.Variable(tf.random_uniform([Nf, Nt], -1, 1), name='w_v')
                tf.summary.histogram("name_scope_test" + '/w', w)
            with tf.name_scope("b"):
                b = tf.Variable(tf.random_uniform([Nt], -1, 1), name='b_v')
                tf.summary.histogram("name_scope_test" + '/b', b)
            yhat = tf.matmul(X, w) + b
            loss = tf.reduce_mean(tf.square(Y - yhat))

            # choose optimizer and operator
            train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

            tf.summary.scalar("wight_max", tf.reduce_mean(w))
            tf.summary.scalar("b_value", tf.reduce_mean(b))
            tf.summary.scalar("X_value", tf.reduce_mean(X))
            merged = tf.summary.merge_all()
            train_summary = tf.summary.FileWriter('./log/tf/', sess.graph)
            # tfmensumm, myadd = sess.run([merged, addAB], feed_dict={: i})

            # init all global var in graph
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            # sess.run(tf.global_variables_initializer())
            for i1 in range(1):
                train_op_r, merged_r, yhat_r = sess.run([train_op, merged, yhat], feed_dict={X: x_data, Y: y_data})
                train_summary.add_summary(merged_r, i1)
            train_summary.close()
            # print("w: {}, b: {}".format(sess.run(w).T, sess.run(b)))
            # for epoch in range(200 * batch_size / M):
            #     i = 0
            #     while i < M:
            #         print("error".center(40, "*"))
            #         print(X.shape)
            #         sess.run(train_op, feed_dict={X: x_data[i: i + batch_size], Y: y_data[i: i + batch_size]})
            #         i += batch_size
            #     print("epoch: {}, w: {}, b: {}".format(epoch, sess.run(w).T, sess.run(b)))

    print("")
    # # %% Let's create some toy data
    # plt.ion()
    # n_observations = 100
    # fig, ax = plt.subplots(1, 1)
    # xs = np.linspace(-3, 3, n_observations)
    # ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
    # ax.scatter(xs, ys)
    # fig.show()
    # plt.draw()
    #
    # # %% tf.placeholders for the input and output of the network. Placeholders are
    # # variables which we need to fill in when we are ready to compute the graph.
    # X = tf.placeholder(tf.float32)
    # Y = tf.placeholder(tf.float32)
    #
    # # %% We will try to optimize min_(W,b) ||(X*w + b) - y||^2
    # # The `Variable()` constructor requires an initial value for the variable,
    # # which can be a `Tensor` of any type and shape. The initial value defines the
    # # type and shape of the variable. After construction, the type and shape of
    # # the variable are fixed. The value can be changed using one of the assign
    # # methods.
    # W = tf.Variable(tf.random_normal([1]), name='weight')
    # b = tf.Variable(tf.random_normal([1]), name='bias')
    # Y_pred = tf.add(tf.multiply(X, W), b)
    #
    # # %% Loss function will measure the distance between our observations
    # # and predictions and average over them.
    # cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)
    #
    # # %% if we wanted to add regularization, we could add other terms to the cost,
    # # e.g. ridge regression has a parameter controlling the amount of shrinkage
    # # over the norm of activations. the larger the shrinkage, the more robust
    # # to collinearity.
    # # cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))
    #
    # # %% Use gradient descent to optimize W,b
    # # Performs a single step in the negative gradient
    # learning_rate = 0.01
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #
    # # %% We create a session to use the graph
    # n_epochs = 1000
    # with tf.Session() as sess:
    #     # Here we tell tensorflow that we want to initialize all
    #     # the variables in the graph so we can use them
    #     sess.run(tf.global_variables_initializer())
    #
    #     # Fit all training data
    #     prev_training_cost = 0.0
    #     for epoch_i in range(n_epochs):
    #         for (x, y) in zip(xs, ys):
    #             sess.run(optimizer, feed_dict={X: x, Y: y})
    #
    #         training_cost = sess.run(
    #             cost, feed_dict={X: xs, Y: ys})
    #         print(training_cost)
    #
    #         if epoch_i % 20 == 0:
    #             ax.plot(xs, Y_pred.eval(
    #                 feed_dict={X: xs}, session=sess),
    #                     'k', alpha=epoch_i / n_epochs)
    #             fig.show()
    #             plt.draw()
    #
    #         # Allow the training to quit if we've reached a minimum
    #         if np.abs(prev_training_cost - training_cost) < 0.000001:
    #             break
    #         prev_training_cost = training_cost
    # fig.show()
    # plt.waitforbuttonpress()
    #
    # with tf.Session() as sess:
    #     # with tf.device():
    #     matrix1 = tf.constant([[3., 3.]])
    #     matrix2 = tf.constant([[2.], [2.]])
    #     product = tf.matmul(matrix1, matrix2)


def tf_auc():
    one = tf.ones_like(label)
    zero = tf.zeros_like(label)
    label = tf.where(label < 0.5, x=zero, y=one)

    auc_value, auc_op = tf.metrics.auc(label_tensor, prediction_tensor, num_thresholds=2000)
    tf.metrics.auc(labels, predictions, weights=None, num_thresholds=200, metrics_collections=None,
                   updates_collections=None, curve='ROC', name=None, summation_method='trapezoidal')

    initializer = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # restore也要重新sess.run(tf.local_variables_initializer())
    # self.saver.restore(sess, latest_ckpt)
    # sess.run(tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(initializer)
        # 要先运行sess.run(auc_op)后再运行计算auc的值
        sess.run(auc_op, feed_dict=feed_dict_t)
        accu = sess.run(auc_value)


def tmp_test():
    # exit(0)
    pass


if __name__ == '__main__':
    # 2. 模块临时测试
    tmp_test()
    # 3. 模型主框架
    label_pd = pd.DataFrame()
    neurous_network(label_pd, batch_size=10)
