# -*- coding:utf-8 -*-
import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


# 文件二，模型常量化
# python 15tensor_freeze2.py --model_folder results

def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    # 输出结点可以看我们模型的定义
    # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    # 所以,output_node_names必须根据不同的网络进行修改
    output_node_names = "Accuracy/predictions"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
    # 相当于将参数已经固化在了图当中
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )
        # print(input_graph_def)
        # print(output_graph_def)

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        # print(output_graph_def.node)
        print("%d ops in the final graph." % len(output_graph_def.node))


def tf2onnx():
    # pip install -U tf2onnx
    # https://blog.csdn.net/u012328159/article/details/81101074
    import tensorflow as tf
    # model_path = "pnet_frozen_model.pb"
    import sys
    model_path = sys.argv[1]
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:  #
        tf.import_graph_def(graph_def, name="")
        for op in graph.get_operations():  #
            print(op.name, op.values())

    with open("graph.proto", "wb") as file:
        graph = tf.get_default_graph().as_graph_def(add_shapes=True)
        file.write(graph.SerializeToString())

    # python -m tf2onnx.convert\
    #     --input tests/models/fc-layers/frozen.pb\
    #     --inputs X:0\
    #     --outputs output:0\
    #     --output tests/models/fc-layers/model.onnx\
    #     --verbose
    # 自定义的
    # python -m tf2onnx.convert --input path/frozen_graph.pb --inputs input_image:0 --outputs cls_prob:0,bbox_pred:0,landmark_pred:0 --output path/pnet.onnx --verbose --custom-ops AdjustContrastv2,AdjustHue,AdjustSaturation

# def tf2onnx():
#     pass

def tf2pb():
    import tensorflow as tf
    import os
    from tensorflow.python.framework import graph_util

    pb_file_path = os.getcwd()

    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, name='x')
        y = tf.placeholder(tf.int32, name='y')
        b = tf.Variable(1, name='b')
        xy = tf.multiply(x, y)
        # 这里的输出需要加上name属性
        op = tf.add(xy, b, name='op_to_store')

        sess.run(tf.global_variables_initializer())

        # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        # 测试 OP
        feed_dict = {x: 10, y: 3}
        print(sess.run(op, feed_dict))

        # 写入序列化的 PB 文件
        with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
            # 输出
            # INFO:tensorflow:Froze 1 variables.
            # Converted 1 variables to const ops.
            # 31


def pb2tf():
    from tensorflow.python.platform import gfile

    sess = tf.Session()
    with gfile.FastGFile(pb_file_path + 'model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 需要有一个初始化的过程
    sess.run(tf.global_variables_initializer())

    # 需要先复原变量
    print(sess.run('b:0'))
    # 1

    # 输入
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
    print(ret)
    # 输出 26


def tf2pb2file():
    import tensorflow as tf
    import os
    from tensorflow.python.framework import graph_util

    pb_file_path = os.getcwd()

    with tf.Session(graph=tf.Graph()) as sess:
        x = tf.placeholder(tf.int32, name='x')
        y = tf.placeholder(tf.int32, name='y')
        b = tf.Variable(1, name='b')
        xy = tf.multiply(x, y)
        # 这里的输出需要加上name属性
        op = tf.add(xy, b, name='op_to_store')

        sess.run(tf.global_variables_initializer())

        # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        # 测试 OP
        feed_dict = {x: 10, y: 3}
        print(sess.run(op, feed_dict))

        # 写入序列化的 PB 文件
        with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # INFO:tensorflow:Froze 1 variables.
        # Converted 1 variables to const ops.
        # 31


        # 官网有误，写成了 saved_model_builder
        builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path + 'savemodel')
        # 构造模型保存的内容，指定要保存的 session，特定的 tag,
        # 输入输出信息字典，额外的信息
        builder.add_meta_graph_and_variables(sess, ['cpu_server_1'])

    # 添加第二个 MetaGraphDef
    # with tf.Session(graph=tf.Graph()) as sess:
    #  ...
    #  builder.add_meta_graph([tag_constants.SERVING])
    # ...

    builder.save()  # 保存 PB 模型


def pb2file2tf():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['cpu_1'], pb_file_path + 'savemodel')
        sess.run(tf.global_variables_initializer())

        input_x = sess.graph.get_tensor_by_name('x:0')
        input_y = sess.graph.get_tensor_by_name('y:0')

        op = sess.graph.get_tensor_by_name('op_to_store:0')

        ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
        print(ret)
        # 只需要指定要恢复模型的 session，模型的 tag，模型的保存路径即可,使用起来更加简单
        # 不知道tensor name的情况下使用呢，实现彻底的解耦呢？ 给add_meta_graph_and_variables方法传入第三个参数，signature_def_map即可


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, help="Model folder to export")
    args = parser.parse_args()

    freeze_graph(args.model_folder)
