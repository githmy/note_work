import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def trainableset():
    # 1. 不会从GraphKeys.TRAINABLE_VARIABLES集合中去除，因此不会影响梯度计算和保存模型
    w1 = tf.stop_gradient(w1)
    # 2.
    trainable_vars = tf.trainable_variables()
    freeze_conv_var_list = [t for t in trainable_vars if not t.name.startswith(u'conv')]
    optimizer = tf.train.AdamOptimizer(lr)
    grads = optimizer.compute_gradients(loss, var_list=freeze_conv_var_list)
    # 3. 梯度截断
    grads, vs = zip(*optimizer.compute_gradients(loss))
    grads, gnorm = tf.clip_by_global_norm(grads, clip)
    self.train_op = optimizer.apply_gradients(zip(grads, vs))


def main():
    v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')
    v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')
    with tf.variable_scope("scope1"):
        v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')
    with tf.name_scope("scope2"):
        v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')

    global_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)
    ema = tf.train.ExponentialMovingAverage(0.99, global_step)

    print("trainable_variables")
    for ele1 in tf.trainable_variables():
        print(ele1.name)

    print()
    print("all_variables")
    for ele2 in tf.all_variables():
        print(ele2.name)


if __name__ == '__main__':
    main()
