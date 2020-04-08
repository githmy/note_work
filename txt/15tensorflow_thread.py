from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


def main():
    # 多线程管理器
    coord = tf.train.Coordinator()
    # 从本地文件里抽取tensor，准备放入FilenameQueue
    tf.train.slice_input_producer()
    # 从文件名队列中提取tensor 多少个线程
    tf.train.batch()
    # 启动入队线程，由多个或单个线程，按照设定规则，把文件读入Filename Queue回线程ID的列表
    tf.train.start_queue_runners()
    # 来启动数据出列和执行计算;
    sess.run()
    # 使用把线程加入主线程
    coord.join(threads)
    # 来发出终止所有线程的命令，
    coord.request_stop()
    # 来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，会抛出一个
    # OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了
    coord.should_stop()

if __name__ == '__main__':
    main()
