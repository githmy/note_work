# coding: utf-8
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import time

seq_len = 25
d = 0.5  # dropout
shape = [4, seq_len, 1]  # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 75
dropout = 0.6
stock_name = '../input/EURUSD/EURUSD1440.csv'

start = time.time()


def get_stock_data(stock_name, inicio='2012', final='2016', normalize=True):
    # df = pd.read_csv(stock_name, names=['Date', 'Open', 'High', 'Low', 'Close'], usecols=[0, 2, 3, 4, 5], index_col=[0])
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


def deep_network():
    # 1. 获取训练数据
    df = get_stock_data(stock_name, inicio='2013', final='2016', normalize=True)
    df_denormalize = get_stock_data(stock_name, inicio='2013', final='2016', normalize=False)
    # 2. 数据转化
    X_train, y_train, X_test, y_test = load_data(df, seq_len, train_split=0.9)
    # 3. 模型创建
    model3 = build_model2(shape, neurons, d)
    # 4. 模型训练
    model3.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)
    # 5. 获取预测数据
    predict_data_norm = datos_predecir()
    predict_data_denorm = datos_predecir(normalise=False)
    # 6. 预测数据转化
    predict_data_length = []
    data = predict_data_norm.as_matrix()
    for index in range(len(data) - seq_len):
        predict_data_length.append(data[index: seq_len + index])
    datos_predecir_array = np.array(predict_data_length)
    # 7. 预测数据
    prediccion = model3.predict(datos_predecir_array)
    end = time.time()
    print(end - start)

    # 8. 绘图
    plt.plot(prediccion)
    plt.plot(np.vstack((np.zeros(seq_len).reshape(-1, 1), prediccion)))
    plt.plot(predict_data_norm['Close'])
    plt.show()


if __name__ == '__main__':
    deep_network()
