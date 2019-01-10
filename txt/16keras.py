# 保存

# Json文件格式
def json_save():
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # 等价于
    json_string = model.get_config()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')


def json_load():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # 另一种方式
    model = model_from_json(open('my_model_architecture.json').read())
    model.load_weights('my_model_weights.h5')


# Yaml文件
def yaml_save():
    # save as YAML
    yaml_string = model.to_yaml()


# HDF5文件
def hdf5_only_weight_save():
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def hdf5_model_weight_save():
    from keras.models import load_model
    model.save('model_weight.h5')  # creates a HDF5 file 'my_model.h5'


def hdf5_only_weight_load():
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")


def hdf5_model_weight_load():
    from keras.models import load_model
    import tensorflow as tf
    from keras import backend as K

    model = load_model('model.h5')

    K.set_learning_phase(0)
    config = model.get_config()
    weights = model.get_weights()
    # **multi-stream network
    multi_stream = 1
    pred = []
    for i in range(multi_stream):
        print(i)
        # output = sess.graph.get_tensor_by_name("dense_1")
        print(model.inputs[i])
        print(model.outputs[i])
        print(model.output.op.name)
        print(model.get_config())
        print(model.get_weights()[0].shape)
        pred[i] = tf.identity(model.outputs[i], name=str(i))
        print(pred[i])


# HDF5文件
def load_weight_diff_layer():
    model.load_weights('my_model_weights.h5', by_name=True)


# 加载 + 变量固化 + 本地保存
def load_varia2constant():
    # 选择性方案
    import os
    import os.path as osp
    import tensorflow as tf
    from keras import backend as K
    from keras.models import load_model
    from tensorflow.python.framework import graph_io
    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph

    """----------------------------------配置路径-----------------------------------"""
    epochs = 20
    h5_model_path = './my_model_ep{}.h5'.format(epochs)
    output_path = '.'
    pb_model_name = 'my_model_ep{}.pb'.format(epochs)

    """----------------------------------导入keras模型------------------------------"""
    K.set_learning_phase(0)
    net_model = load_model(h5_model_path)

    print('input is :', net_model.input.name)
    print('output is:', net_model.output.name)

    """----------------------------------保存为.pb格式------------------------------"""
    sess = K.get_session()
    frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
    graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)


# 加载 + 变量固化 + 本地保存
def keras_migration():
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(64, activation='relu', W_regularizer=EigenvalueRegularizer(10)))
    top_model.add(Dense(10, activation='softmax', W_regularizer=EigenvalueRegularizer(10)))
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)


# 结构查看
def print_struture():
    model.summary()
    from keras.utils.visualize_util import plot
    plot(model, to_file='model1.png', show_shapes=True)


# 学习率
def learn_rate():
    import keras.backend as K
    from keras.callbacks import LearningRateScheduler
    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)
    model.fit(train_x, train_y, batch_size=32, epochs=5, callbacks=[reduce_lr])

# 元素转层
def var2layer():
    from keras import backend as K
    from keras.layers import Lambda

    def var_trans(input):
        a, b = input
        return a + b

    vartt = K.random_uniform_variable(shape=(1, 5), low=-1, high=1, dtype="float32")
    x_ex_in = Lambda(var_trans, name='varlay')([vartt, vartt])


if __name__ == '__main__':
    pass
