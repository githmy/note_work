# coding: utf-8

# In[1]:


# I try to predict values not used in a model to predict results.
# Internet is plenty of models using X_test to predict.
# However, I havent seen a lot of models using data outside the model
# Thank you


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import time
import keras
import tensorflow as tf
from tensorflow.python.util import compat
from keras import backend as K
import os


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
seq_len = 25
d = 0.5  # dropout
shape = [4, seq_len, 1]  # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 75
dropout = 0.6
stock_name = '../input/EURUSD/EURUSD1440.csv'

start = time.time()

def gpu_set():
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    # 设置session
    KTF.set_session(sess)

def keras2pb():
    def export_savedmodel(model,pb_path):
        '''
        传入keras model会自动保存为pb格式
        '''
        model_path = "model/"  # 模型保存的路径
        model_version = 0  # 模型保存的版本
        # 从网络的输入输出创建预测的签名
        model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'input': model.input}, outputs={'output': model.output})
        # 使用utf-8编码将 字节或Unicode 转换为字节
        export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))  # 将保存路径和版本号join
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)  # 生成"savedmodel"协议缓冲区并保存变量和模型
        builder.add_meta_graph_and_variables(  # 将当前元图添加到savedmodel并保存变量
            sess=K.get_session(),  # 返回一个 session 默认返回tf的sess,否则返回keras的sess,两者都没有将创建一个全新的sess返回
            tags=[tf.saved_model.tag_constants.SERVING],  # 导出模型tag为SERVING(其他可选TRAINING,EVAL,GPU,TPU)
            clear_devices=True,  # 清除设备信息
            signature_def_map={  # 签名定义映射
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:  # 默认服务签名定义密钥
                    model_signature  # 网络的输入输出策创建预测的签名
            })
        builder.save()  # 将"savedmodel"协议缓冲区写入磁盘.
        print("save model pb success ...")

    model = keras.models.load_model('model_data/weight.h5')  # 加载已训练好的.h5格式的keras模型
    export_savedmodel(model)  # 将模型传入保存模型的方法内,模型保存成功.

def h5_2pb():
    def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
        if os.path.exists(output_dir) == False:
            os.mkdir(output_dir)
        out_nodes = []
        for i in range(len(h5_model.outputs)):
            out_nodes.append(out_prefix + str(i + 1))
            tf.identity(h5_model.output[i], out_prefix + str(i + 1))
        sess = K.get_session()
        from tensorflow.python.framework import graph_util, graph_io
        init_graph = sess.graph.as_graph_def()
        main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
        graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
        if log_tensorboard:
            from tensorflow.python.tools import import_pb_to_tensorboard
            import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)

    # 输出路径
    output_dir = os.path.join(os.getcwd(), "trans_model")
    # 加载模型
    print(weight_file_path)
    h5_model = load_model(weight_file_path)
    # model.load_weights
    h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
    print('model saved')


def graph_manipulate():
    g3 = tf.get_default_graph()
    with g3.as_default():
        basemodel.load_weights(modelPath)

def get_stock_data(stock_name, inicio='2012', final='2016', normalize=True):
    df = pd.read_csv(stock_name, names=['Date', 'Open', 'High', 'Low', 'Close'], usecols=[0, 2, 3, 4, 5], index_col=[0])
    df = df.loc[inicio:final]

    df.reset_index(inplace=True, drop=True)
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return df


# Formulas to make LSTM model
def load_data(stock, seq_len, train_split=0.9):
    amount_of_features = len(stock.columns)
    print(stock.columns)
    print(seq_len)
    data = stock.as_matrix()
    sequence_length = seq_len + 1  # index starting from 0 esto se hace ya que el indice empieza por 0 entonces quedaria

    result = []
    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days
    result = np.array(result)
    row = round(train_split * result.shape[0])  # 90% split

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    print(len(data))
    print(result.shape)
    # print(result[0:2])
    print(X_train.shape)
    # print(X_train[0:2])
    print(y_train.shape)
    # print(y_train[0:2])
    print(X_test.shape)
    # print(X_test[0:2])
    print(y_test.shape)
    # print(y_test[0:2])
    return [X_train, y_train, X_test, y_test]


def build_model2(layers, neurons, d):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


df = get_stock_data(stock_name, inicio='2013', final='2016', normalize=True)
df_denormalize = get_stock_data(stock_name, inicio='2013', final='2016', normalize=False)

X_train, y_train, X_test, y_test = load_data(df, seq_len, train_split=0.9)
model3 = build_model2(shape, neurons, d)
model3.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)


# Making predictions with new data not used in the model
def datos_predecir(file='../input/EURUSD/EURUSD_2017.csv', normalise=True):
    df = pd.read_csv(file, usecols=[1, 2, 3, 4, 5], index_col=[0])
    df.reset_index(inplace=True, drop=True)
    if normalise == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return df


predict_data_norm = datos_predecir()
predict_data_denorm = datos_predecir(normalise=False)

predict_data_length = []
data = predict_data_norm.as_matrix()
for index in range(len(data) - seq_len):
    predict_data_length.append(data[index: seq_len + index])

datos_predecir = np.array(predict_data_length)
prediccion = model3.predict(datos_predecir)

end = time.time()
print(end - start)
plt.plot(prediccion)
plt.plot(np.vstack((np.zeros(seq_len).reshape(-1, 1), prediccion)))
plt.plot(predict_data_norm['Close'])
plt.show()
