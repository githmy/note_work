from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np
import keras

# 多分支 示例
# construct model
main_input = Input((100,), dtype='int32', name='main_input')

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

aux_input = Input((5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, aux_input])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.3})

# train data
samples_len = 300
main_data = np.random.randint(1, 10000, size=(samples_len, 100), dtype='int32')
aux_data = np.random.randint(0, 10, size=(samples_len, 5), dtype='int32')
main_labels = np.random.randint(0, 2, size=(samples_len, 1), dtype='int32')

model.fit(x={'main_input': main_data, 'aux_input': aux_data},
          y={'main_output': main_labels, 'aux_output': main_labels},
          batch_size=32, epochs=10, verbose=1)

score = model.evaluate(x={'main_input': main_data, 'aux_input': aux_data},
                       y={'main_output': main_labels, 'aux_output': main_labels},
                       batch_size=10, verbose=1)
print(score)
