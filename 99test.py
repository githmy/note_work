from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, \
    Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from random import choices
import kerastuner as kt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import os, gc, random, datetime
import xgboost as xgb
import datatable as dtable
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from joblib import dump, load
import xgboost as xgb
import optuna
from time import time
from numba import njit
from tqdm.notebook import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import joblib
import janestreet

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)

plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [14, 8]  # width, height
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [20, 12]  # width, height
set_config(display='diagram')

# transform
tf.random.set_seed(42)
device = 'GPU' if 'GPU' in tf.test.gpu_device_name() else 'CPU/TPU'
print('Device:', device)

if device == 'GPU':
    import cudf
    import cupy as cp

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()  # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

MIXED_PRECISION = False
XLA_ACCELERATE = True

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    if tpu:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')

# # Preprocessing

print('Loading...')
train = dtable.fread('../input/jane-street-market-prediction/train.csv').to_pandas()
features = [c for c in train.columns if 'feature' in c]

print('Filling...')
train = train.query('weight > 0').reset_index(drop=True)
train[features] = train[features].fillna(method='ffill').fillna(0)
train['action'] = (train['resp'] > 0).astype('int')

print('Finish.')


# # Training
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='swish'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        #         self.pos_encoding = positional_encoding(self.maximum_position_encoding,
        #                                                 self.d_model)
        #         self.embedding = tf.keras.layers.Dense(self.d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maximum_position_encoding,
                                                 output_dim=self.d_model)

        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout': self.dropout,
        })
        return config

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        #         x += self.pos_encoding[:, :seq_len, :]
        #         x = self.embedding(x)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x += self.pos_emb(positions)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


def create_transformer_model(num_columns, num_labels, num_layers, d_model, num_heads, dff, window_size, dropout_rate,
                             weight_decay, label_smoothing, learning_rate):
    inp = tf.keras.layers.Input(shape=(window_size, num_columns))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
    x = TransformerEncoder(num_layers, d_model, num_heads, dff, window_size, dropout_rate)(x)
    out = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x[:, -1, :])

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tfa.optimizers.AdamW(weight_decay=weight_decay, learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
                  metrics=tf.keras.metrics.AUC(name='AUC'),
                  )
    return model


batch_size = 4096 * strategy.num_replicas_in_sync
num_layers = 1
d_model = 96
num_heads = 1
dff = 64
window_size = 3
dropout_rate = 0.15
weight_decay = 0
label_smoothing = 1e-2
learning_rate = 1e-3 * strategy.num_replicas_in_sync
verbose = 1

with strategy.scope():
    model = create_transformer_model(len(features), 1, num_layers, d_model, num_heads, dff, window_size, dropout_rate,
                                     weight_decay, label_smoothing, learning_rate)
model.summary()

K.clear_session()
del model
rubbish = gc.collect()


# Use Tensorflow Dataset
def prepare_dataset(X, y, window_size, batch_size, mode='training'):
    x_ds = tf.data.Dataset.from_tensor_slices(X)
    y_ds = tf.data.Dataset.from_tensor_slices(y[window_size - 1:])
    x_ds = x_ds.window(window_size, shift=1, drop_remainder=True)
    x_ds = x_ds.flat_map(lambda window: window.batch(window_size))
    dataset = tf.data.Dataset.zip((x_ds, y_ds))
    if mode == 'training':
        buffer_size = batch_size * 8
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)  # , drop_remainder = True
    elif mode == 'validation':
        dataset = dataset.batch(batch_size)
        dataset = dataset.cache()
    elif mode == 'testing':
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset


# Use Numpy [may cause Out-of-Memory (OOM) error]
def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.squeeze(np.lib.stride_tricks.as_strided(a, shape=s, strides=strides), axis=1)


# # Train-Test-Split Training
X_tr = train.loc[train['date'] < 303, features].values
y_tr = train.loc[train['date'] < 303, 'action'].values

X_tr2 = train.loc[(train['date'] >= 303) & (train['date'] <= 367), features].values
y_tr2 = train.loc[(train['date'] >= 303) & (train['date'] <= 367), 'action'].values

X_val = train.loc[train['date'] > 367, features].values
y_val = train.loc[train['date'] > 367, 'action'].values

rubbish = gc.collect()

X_tr = rolling_window(X_tr, (window_size, len(features)))
X_val = rolling_window(X_val, (window_size, len(features)))
y_tr = y_tr[window_size - 1:]
y_val = y_val[window_size - 1:]
X_tr2 = rolling_window(X_tr2, (window_size, len(features)))
y_tr2 = y_tr2[window_size - 1:]

# Train on the training-1 set and validate on the validation set.
start_time_fold = time()
ckp_path = 'JSTransformer.hdf5'
with strategy.scope():
    model = create_transformer_model(len(features), 1, num_layers, d_model, num_heads, dff, window_size, dropout_rate,
                                     weight_decay, label_smoothing, learning_rate)
rlr = ReduceLROnPlateau(monitor='val_AUC', factor=0.1, patience=3, verbose=verbose,
                        min_delta=1e-4, mode='max')
ckp = ModelCheckpoint(ckp_path, monitor='val_AUC', verbose=0,
                      save_best_only=True, save_weights_only=True, mode='max')
es = EarlyStopping(monitor='val_AUC', min_delta=1e-4, patience=7, mode='max',
                   baseline=None, restore_best_weights=True, verbose=0)
history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=batch_size,
                    epochs=1000, callbacks=[rlr, ckp, es], verbose=verbose)
hist = pd.DataFrame(history.history)
print(f'[{str(datetime.timedelta(seconds = time() - start_time_fold))[0:7]}] ROC AUC:\t', hist['val_AUC'].max())

K.clear_session()
del model, X_tr, y_tr
rubbish = gc.collect()

# Check the best number of epochs for finetuning.
start_time_fold = time()

with strategy.scope():
    model = create_transformer_model(len(features), 1, num_layers, d_model, num_heads, dff,
                                     window_size, dropout_rate, weight_decay, label_smoothing,
                                     learning_rate / 100)
model.load_weights(ckp_path)
es = EarlyStopping(monitor='val_AUC', min_delta=1e-4, patience=7, mode='max',
                   baseline=None, restore_best_weights=True, verbose=0)
history2 = model.fit(X_tr2, y_tr2, validation_data=(X_val, y_val), batch_size=batch_size,
                     epochs=1000, callbacks=[es], verbose=verbose)
hist2 = pd.DataFrame(history2.history)
print(f'[{str(datetime.timedelta(seconds = time() - start_time_fold))[0:7]}] ROC AUC:\t', hist2['val_AUC'].max())
finetune_epochs = hist2['val_AUC'].argmax() + 1

K.clear_session()
del model
rubbish = gc.collect()

# Train on both training-2 and validation sets.
with strategy.scope():
    model = create_transformer_model(len(features), 1, num_layers, d_model, num_heads, dff,
                                     window_size, dropout_rate, weight_decay, label_smoothing,
                                     learning_rate / 100)
model.load_weights(ckp_path)
model.fit(np.concatenate((X_tr2, X_val)), np.concatenate((y_tr2, y_val)),
          batch_size=batch_size, epochs=finetune_epochs, verbose=verbose)
model.save_weights(ckp_path)

K.clear_session()
del model, X_tr2, y_tr2, X_val, y_val
rubbish = gc.collect()

# # GroupCV Training
# gkf = GroupKFold(n_splits = 5)
# for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
#     start_time_fold = time()
#     X_tr, X_val = train.loc[tr, features].values, train.loc[te, features].values
#     y_tr, y_val = train.loc[tr, 'action'].values, train.loc[te, 'action'].values
#     train_steps = int((X_tr.shape[0] // batch_size) + 1)
#     val_steps = int((X_val.shape[0] // batch_size) + 1)
#     dataset_tr = prepare_dataset(X_tr, y_tr, window_size, batch_size, 'training')
#     dataset_val = prepare_dataset(X_val, y_val, window_size, batch_size, 'validation')
#     ckp_path = f'JSModel_{fold}.hdf5'
#     with strategy.scope():
#         model = create_transformer_model(len(features), 1, num_layers, d_model, num_heads, dff, window_size, dropout_rate, weight_decay, label_smoothing, learning_rate)
#     rlr = ReduceLROnPlateau(monitor = 'val_AUC', factor = 0.1, patience = 3, verbose = verbose,
#                             min_delta = 1e-4, mode = 'max')
#     ckp = ModelCheckpoint(ckp_path, monitor = 'val_AUC', verbose = 0,
#                           save_best_only = True, save_weights_only = True, mode = 'max')
#     es = EarlyStopping(monitor = 'val_AUC', min_delta = 1e-4, patience = 7, mode = 'max',
#                        baseline = None, restore_best_weights = True, verbose = 0)
#     history = model.fit(dataset_tr, steps_per_epoch = train_steps,
#                         validation_data = dataset_val, validation_steps = val_steps,
#                         epochs = 1000, callbacks = [rlr, ckp, es], verbose = verbose)
#     hist = pd.DataFrame(history.history)
#     K.clear_session()
#     del model
#     rubbish = gc.collect()
#     # Finetune 3 epochs on validation set with small learning rate
#     with strategy.scope():
#         model = create_transformer_model(len(features), 1, num_layers, d_model, num_heads, dff, window_size, dropout_rate, weight_decay, label_smoothing, learning_rate / 100)
#     model.load_weights(ckp_path)
#     dataset_val = prepare_dataset(X_val, y_val, window_size, batch_size, 'training')
#     model.fit(dataset_val, steps_per_epoch = val_steps, epochs = 3, verbose = 0)
#     model.save_weights(ckp_path)
#     print(f'[{str(datetime.timedelta(seconds = time() - start_time_fold))[0:7]}] Fold {fold} ROC AUC:\t', hist['val_AUC'].max())
#     K.clear_session()
#     del model, X_tr, X_val, y_tr, y_val, dataset_tr, dataset_val
#     rubbish = gc.collect()
#     break

# # Load Model
with strategy.scope():
    model = create_transformer_model(len(features), 1, num_layers, d_model, num_heads, dff, window_size, dropout_rate,
                                     weight_decay, label_smoothing, learning_rate)
model.load_weights('./JSTransformer.hdf5')


# # Submitting
@njit
def fast_fillna(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


train.loc[0, features[1:]] = fast_fillna(train.loc[0, features[1:]].values, 0)

env = janestreet.make_env()
env_iter = env.iter_test()

opt_th = 0.505
tmp = np.zeros((1, window_size, len(features)))
for (test_df, pred_df) in tqdm(env_iter):
    if test_df['weight'].item() > 0:
        x_tt = test_df.loc[:, features].values
        x_tt[0] = fast_fillna(x_tt[0], tmp[0, -1])
        tmp[0] = np.concatenate((tmp[0, 1:], x_tt))
        pred = model(tmp, training=False).numpy().item()
        pred_df.action = np.where(pred >= opt_th, 1, 0).astype(int)
    else:
        pred_df.action = 0
    env.predict(pred_df)

train = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv', nrows=30000)

features = [col for col in list(train.columns) if 'feature' in col]
train = train[train['weight'] != 0]
train['action'] = (train['resp'].values > 0).astype(int)
f_mean = train.mean()
train = train.fillna(f_mean)
X = train.loc[:, features]
y = train.loc[:, 'action']
del train
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## voting
classifiers = [['Neural Network :', MLPClassifier(max_iter=1000)],
               ['LogisticRegression :', LogisticRegression(max_iter=1000)],
               ['ExtraTreesClassifier :', ExtraTreesClassifier()],
               ['DecisionTree :', DecisionTreeClassifier()],
               ['RandomForest :', RandomForestClassifier()],
               ['Naive Bayes :', GaussianNB()],
               ['KNeighbours :', KNeighborsClassifier()],
               ['SVM :', SVC()],
               ['AdaBoostClassifier :', AdaBoostClassifier()],
               ['GradientBoostingClassifier: ', GradientBoostingClassifier()],
               ['XGB :', XGBClassifier()],
               ['CatBoost :', CatBoostClassifier(logging_level='Silent')]]

predictions_df = pd.DataFrame()
predictions_df['action'] = y_test

for name, classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train.ravel())
    predictions = classifier.predict(X_test)
    predictions_df[name.strip(" :")] = predictions
    print(name, accuracy_score(y_test, predictions))

clf1 = ExtraTreesClassifier()
clf2 = CatBoostClassifier(logging_level='Silent')
clf3 = RandomForestClassifier()
eclf1 = VotingClassifier(estimators=[('ExTrees', clf1), ('CatBoost', clf2), ('RF', clf3)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print(accuracy_score(y_test, predictions))

c = []
c.append(cross_val_score(clf1, X_train, y_train, scoring='accuracy', cv=10).mean())
c.append(cross_val_score(clf2, X_train, y_train, scoring='accuracy', cv=10).mean())
c.append(cross_val_score(clf3, X_train, y_train, scoring='accuracy', cv=10).mean())
print(c)

# ### Tabnet is like sklearn for deeplearning

clf = TabNetClassifier()
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], max_epochs=2  # Change this to increase the accuracy
)

env = janestreet.make_env()  # initialize the environment
iter_test = env.iter_test()  # an iterator which loops over the test set

for (test_df, prediction_df) in env.iter_test():
    test = test_df.fillna(f_mean)
    X_test = test.loc[:, features]
    X_test = np.array(X_test)
    y_preds = clf.predict(X_test)
    prediction_df.action = y_preds
    env.predict(prediction_df)


# MLP ， 如果 `feature_0=1` 且 `resp_hat*feature_0 > th`.


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.max_test_group_size = max_test_group_size
        self.group_gap = group_gap
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(("Cannot have number of folds={0} greater than"
                              " the number of groups={1}").format(n_folds, n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size, n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(np.concatenate((train_array, train_array_tmp)), axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start: group_test_start + group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(np.concatenate((test_array, test_array_tmp)), axis=None), axis=None)

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


class PurgedGroupTimeSeriesSplitStacking(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    stacking_mode : bool, default=True
        Whether to provide an additional set to test a stacking classifier or not. 
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    max_val_group_size : int, default=Inf
        Maximum group size for a single validation set.
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split, if stacking_mode = True and None 
        it defaults to max_val_group_size.
    val_group_gap : int, default=None
        Gap between train and validation
    test_group_gap : int, default=None
        Gap between validation and test, if stacking_mode = True and None 
        it defaults to val_group_gap.
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 stacking_mode=True,
                 max_train_group_size=np.inf,
                 max_val_group_size=np.inf,
                 max_test_group_size=np.inf,
                 val_group_gap=None,
                 test_group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.max_val_group_size = max_val_group_size
        self.max_test_group_size = max_test_group_size
        self.val_group_gap = val_group_gap
        self.test_group_gap = test_group_gap
        self.verbose = verbose
        self.stacking_mode = stacking_mode

    def split(self, X, y=None, groups=None):
        if self.stacking_mode:
            return self.split_ensemble(X, y, groups)
        else:
            return self.split_standard(X, y, groups)

    def split_standard(self, X, y=None, groups=None):
        """Generate indices to split data into training and validation set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/validation set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        val : ndarray
            The validation set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        group_gap = self.val_group_gap
        max_val_group_size = self.max_val_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_val_size = min(n_groups // n_folds, max_val_group_size)
        group_val_starts = range(n_groups - n_splits * group_val_size,
                                 n_groups, group_val_size)
        for group_val_start in group_val_starts:
            train_array = []
            val_array = []

            group_st = max(0, group_val_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_val_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for val_group_idx in unique_groups[group_val_start:
                    group_val_start +
                    group_val_size]:
                val_array_tmp = group_dict[val_group_idx]
                val_array = np.sort(np.unique(
                    np.concatenate((val_array,
                                    val_array_tmp)),
                    axis=None), axis=None)

            val_array = val_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in val_array]

    def split_ensemble(self, X, y=None, groups=None):
        """Generate indices to split data into training, validation and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        val : ndarray
            The validation set indices for that split (testing indices for base classifiers).
        test : ndarray
            The testing set indices for that split (testing indices for final classifier)
        """

        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")

        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        val_group_gap = self.val_group_gap
        test_group_gap = self.test_group_gap
        if test_group_gap is None:
            test_group_gap = val_group_gap
        max_train_group_size = self.max_train_group_size
        max_val_group_size = self.max_val_group_size
        max_test_group_size = self.max_test_group_size
        if max_test_group_size is None:
            max_test_group_size = max_val_group_size

        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)

        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_val_size = min(n_groups // n_folds, max_val_group_size)
        group_test_size = min(n_groups // n_folds, max_test_group_size)

        group_test_starts = range(n_groups - n_splits * group_test_size, n_groups, group_test_size)
        train_indices = []
        val_indices = []
        test_indices = []

        for group_test_start in group_test_starts:

            train_array = []
            val_array = []
            test_array = []

            val_group_st = max(max_train_group_size + val_group_gap,
                               group_test_start - test_group_gap - max_val_group_size)

            train_group_st = max(0, val_group_st - val_group_gap - max_train_group_size)

            for train_group_idx in unique_groups[train_group_st:(val_group_st - val_group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for val_group_idx in unique_groups[val_group_st:(group_test_start - test_group_gap)]:
                val_array_tmp = group_dict[val_group_idx]
                val_array = np.sort(np.unique(
                    np.concatenate((val_array,
                                    val_array_tmp)),
                    axis=None), axis=None)

            val_array = val_array[val_group_gap:]

            for test_group_idx in unique_groups[group_test_start:(group_test_start + group_test_size)]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)

            test_array = test_array[test_group_gap:]

            yield [int(i) for i in train_array], [int(i) for i in val_array], [int(i) for i in test_array]


class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, X, y, splits, batch_size=32, epochs=1, callbacks=None):
        val_losses = []
        for train_indices, test_indices in splits:
            X_train, X_test = [x[train_indices] for x in X], [x[test_indices] for x in X]
            y_train, y_test = [a[train_indices] for a in y], [a[test_indices] for a in y]
            if len(X_train) < 2:
                X_train = X_train[0]
                X_test = X_test[0]
            if len(y_train) < 2:
                y_train = y_train[0]
                y_test = y_test[0]

            model = self.hypermodel.build(trial.hyperparameters)
            hist = model.fit(X_train, y_train,
                             validation_data=(X_test, y_test),
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=callbacks)

            val_losses.append([hist.history[k][-1] for k in hist.history])
        val_losses = np.asarray(val_losses)
        self.oracle.update_trial(trial.trial_id,
                                 {k: np.mean(val_losses[:, i]) for i, k in enumerate(hist.history.keys())})
        self.save_model(trial.trial_id, model)


# ### Loading the training data

TRAINING = False
USE_FINETUNE = True
FOLDS = 5
SEED = 42

train = pd.read_csv('../input/jane-street-market-prediction/train.csv')
train = train.query('date > 85').reset_index(drop=True)
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns})  # limit memory use
train.fillna(train.mean(), inplace=True)

train['action'] = ((train['resp_1'] > 0.00001) & (train['resp_2'] > 0.00001) & (train['resp_3'] > 0.00001) & (
    train['resp_4'] > 0.00001) & (train['resp'] > 0.00001)).astype('int')
features = [c for c in train.columns if 'feature' in c]
resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']
EPSILON = {c: 0.0 for c in resp_cols}

X = train[features].values
y = train['resp'].values
y = (y * train[features[0]].values).reshape((-1, 1))
y = np.stack([(train[c] > EPSILON[c]).astype('int') for c in resp_cols]).T  # Multitarget
# feature_0 is buy/sell order (we don't know which but it does not matter), and we convert resp accordingly.
weights = train['weight'].values
resps = train['resp'].values
dates = train['date'].values
f_mean = np.mean(X, axis=0)
f_mean = np.mean(train[features[1:]].values, axis=0)

f0s = train['feature_0'].values

del train


def create_encoder(input_dim, output_dim, noise=0.05):
    i = Input(input_dim)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)

    encoded = Dense(64, activation='relu')(encoded)

    decoded = BatchNormalization()(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, name='decoded')(decoded)

    x = Dense(32)(decoded)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.2)(x)
    x = Dense(16)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='sigmoid', name='label_output')(x)

    encoder = Model(inputs=i, outputs=decoded)
    autoencoder = Model(inputs=i, outputs=[decoded, x])

    autoencoder.compile(optimizer=Adam(0.01), loss={'decoded': 'mse', 'label_output': 'binary_crossentropy'})
    return autoencoder, encoder


# ### Creating the MLP.
def create_model(input_dim, output_dim):
    inputs = Input(input_dim)
    x = BatchNormalization()(inputs)
    x = Dropout(0.2)(x)

    for i in range(3):
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.swish)(x)
        x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='linear')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(0.001), loss='logcosh', metrics=['mae', 'mse'])
    return model


def utility_score_bincount(date, weight, resp, action):
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u


class LiteModel:
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]


val_metric = 'val_loss'
val_direction = 'min'
EPOCHS = 1000
BATCH_SIZE = 4096

FOLDS = 5
SEED = 42
SEEDS = [1, 42, 123]

if TRAINING:
    oof = np.zeros((len(SEEDS), *y.shape))
    for j, SEED in enumerate(SEEDS):
        set_all_seeds(SEED)

        autoencoder, encoder = create_encoder(X.shape[-1], y.shape[-1], noise=0.1)
        autoencoder.fit(X, (X, y),
                        epochs=1000,
                        batch_size=4096,
                        validation_split=0.1,
                        callbacks=[EarlyStopping('val_loss', patience=10, restore_best_weights=True)])
        encoder.save_weights(f'./encoder_{SEED}.hdf5')

        model_fn = lambda hp: create_model(hp, X.shape[-1], y.shape[-1], encoder)

        tuner = CVTuner(
            hypermodel=model_fn,
            oracle=kt.oracles.BayesianOptimization(
                objective=kt.Objective('val_loss', direction='min'),
                num_initial_points=4,
                max_trials=20,
                project_name=f'jane_street_{SEED}',
                seed=SEED)
        )

        gkf = PurgedGroupTimeSeriesSplit(n_splits=FOLDS, group_gap=31)
        splits = list(gkf.split(y, groups=train['date'].values))
        tuner.search((X,), (y,), splits=splits, batch_size=4096, epochs=100,
                     callbacks=[EarlyStopping('val_loss', patience=5),
                                ReduceLROnPlateau('val_loss', patience=3)])
        hp = tuner.get_best_hyperparameters(1)[0]
        oof = np.zeros(y.shape)
        pd.to_pickle(hp, f'./best_hp_{SEED}.pkl')
        for fold, (train_indices, test_indices) in enumerate(splits):
            model = model_fn(hp)
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=4096,
                      callbacks=[EarlyStopping('val_loss', patience=10, restore_best_weights=True),
                                 ReduceLROnPlateau('val_loss', patience=5)])
            oof[j, test_indices] += model.predict(X_test)
            model.save_weights(f'./model_{SEED}_{fold}.hdf5')
            model.compile(Adam(hp.get('lr') / 100),
                          loss=BinaryCrossentropy(
                              label_smoothing=10 * hp.get('label_smoothing')))  # trying something with ls here
            model.fit(X_test, y_test, epochs=3, batch_size=4096)
            model.save_weights(f'./model_{SEED}_{fold}_finetune.hdf5')
else:
    models = []
    for SEED in SEEDS:
        _, encoder = create_encoder(X.shape[-1], y.shape[-1], noise=0.1)
        encoder.load_weights(f'../input/jsautoencoder/encoder_{SEED}.hdf5')
        encoder.trainable = False

        model_fn = lambda hp: create_model(hp, X.shape[-1], y.shape[-1], encoder)
        hp = pd.read_pickle(f'../input/jsautoencoder/best_hp_{SEED}.pkl')
        for f in range(FOLDS):
            model = model_fn(hp)
            if USE_FINETUNE:
                model.load_weights(f'../input/jsautoencoder/model_{SEED}_{f}_finetune.hdf5')
            else:
                model.load_weights(f'../input/jsautoencoder/model_{SEED}_{f}.hdf5')
            model = LiteModel.from_keras_model(model)
            models.append(model)

# best_th
if TRAINING:
    for j in range(len(SEEDS)):
        ypred = np.mean(oof[j], axis=-1)
        pmin = ypred.min()
        pmax = ypred.max()
        r = pmax - pmin
        best_us = 0
        best_th = pmin
        for th in tqdm(np.arange(pmin, pmax, r / 1000)):
            ypred = np.where(ypred > th, 1, 0)
            us = utility_score_bincount(train['date'].values, train['weight'].values, train['resp'].values, ypred)
            if us > best_us:
                best_us = us
                best_th = th
        print(best_us, best_th)

best_th = 0.5

if not TRAINING:
    env = janestreet.make_env()
    th = best_th
    w = np.asarray([0.1, 0.1, 0.1, 0.5, 0.2])
    for (test_df, pred_df) in tqdm(env.iter_test()):
        w = test_df['weight'].item()
        if w > 0:
            f0 = test_df['feature_0'].item()
            x_tt = test_df.loc[:, features[1:]].values
            if np.isnan(x_tt.sum()):
                x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt) * f_mean
            pred = np.mean([model.predict(x_tt) for model in models], axis=0)
            pred_df.action = np.where(pred * f0 > th, 1, 0).astype(int)
        else:
            pred_df.action = 0
        env.predict(pred_df)

# xboost
n_samples = 2000
n_groups = 20
assert n_samples % n_groups == 0

idx = np.linspace(0, n_samples - 1, num=n_samples)
X_train = np.random.random(size=(n_samples, 5))
y_train = np.random.choice([0, 1], n_samples)
groups = np.repeat(np.linspace(0, n_groups - 1, num=n_groups), n_samples / n_groups)


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    cmap_cv = plt.cm.coolwarm
    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)  # inplace
    cmap_data = ListedColormap(jet(seq))
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


fig, ax = plt.subplots()
cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=7,
    group_gap=2,
    max_test_group_size=3
)

plot_cv_indices(cv, X_train, y_train, groups, ax, 5, lw=20)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


train_data = (
    dt.fread('../input/jane-street-market-prediction/train.csv')
        .to_pandas()
        .query('weight > 0')
        .pipe(reduce_mem_usage)
)

feature_names = train_data.columns[train_data.columns.str.contains('feature')]

fig, ax = plt.subplots()
cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=15,
    group_gap=5,
    max_test_group_size=5
)

plot_cv_indices(
    cv,
    train_data.query('date < 50')[
        train_data.columns[train_data.columns.str.contains('feature')]
    ].values,
    (train_data.query('date < 50')['resp'] > 0).astype(int).values,
    train_data.query('date < 50')['date'].values,
    ax,
    5,
    lw=20
)

# TODO: feature_0 should not be scaled
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = StandardScaler()

logistic = LogisticRegression(
    max_iter=1000,
    tol=0.1,
    verbose=10,
    penalty='l1',
    solver='liblinear',
    random_state=42
)

pipe = Pipeline(steps=[
    ('imputer', imp_mean),
    ('scaler', scaler),
    ('logistic', logistic)
])
train_data.info()

param_grid = {
    'logistic__C': np.logspace(-3, 1.5, 7),  # lower C is more regularization
}

scoring = {'AUC': 'roc_auc'}

cv = PurgedGroupTimeSeriesSplit(
    n_splits=3,
    max_train_group_size=150,
    group_gap=20,
    max_test_group_size=60
)

search = GridSearchCV(
    pipe,
    param_grid,
    n_jobs=3,
    cv=cv,
    verbose=10,
    scoring=scoring,
    refit=False,  # 'AUC',   # <-- do we want to refit on the entire dataset?
    return_train_score=True
)

gc.collect()
FIT = True

if FIT:
    search.fit(
        train_data[
            train_data.columns[train_data.columns.str.contains('feature')]
        ].values,
        (train_data['resp'] > 0).astype(int).values,
        groups=train_data['date'].values,
    )

results = search.cv_results_

param = 'param_' + list(param_grid.keys())[0]

plt.figure(figsize=(12, 6))
plt.title("GridSearchCV", fontsize=16)
plt.xlabel(list(param_grid.keys())[0])
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0.00075, 40)
ax.set_ylim(0.45, 0.55)
ax.set_xscale('log')

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results[param].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.4f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()

results_idx = np.argmax(results['mean_test_AUC'])
best_param = results[param][results_idx]

print(f'The best setting for C is {best_param}')

# ## Fit the Best Estimator on the Entire Data
logistic = LogisticRegression(
    max_iter=1000,
    C=0.1,
    tol=0.1,
    verbose=10,
    penalty='l1',  # <-- L1 norm to enforce sparsity of coefficients
    solver='liblinear'
)

pipe_lr = Pipeline(steps=[
    ('imputer', imp_mean),
    ('scaler', scaler),
    ('logistic', logistic)
])

gc.collect()

pipe_lr.fit(
    train_data.query('date > 100')[
        train_data.columns[train_data.columns.str.contains('feature')]
    ].values,
    (train_data.query('date > 100')['resp'] > 0).astype(int).values
)

gc.collect()

y_labels = (train_data['resp'] > 0).astype(int).values
X_train = train_data[
    train_data.columns[train_data.columns.str.contains('feature')]
].values
groups = train_data['date'].values

cv = PurgedGroupTimeSeriesSplit(
    n_splits=3,
    max_train_group_size=150,
    group_gap=20,
    max_test_group_size=60
)


def objective(trial, cv=cv, cv_fold_func=np.average):
    # Optuna suggest params
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90),
        'gamma': trial.suggest_int('gamma', 0, 20),
        'missing': -999,
        'tree_method': 'gpu_hist'
    }

    # setup the pieline
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()
    clf = xgb.XGBClassifier(**params)

    pipe = Pipeline(steps=[
        ('imputer', imp_mean),
        ('scaler', scaler),
        ('xgb', clf)
    ])

    # fit for all folds and return composite AUC score
    aucs = []
    for i, (train_idx, valid_idx) in enumerate(cv.split(
            X_train,
            y_labels,
            groups=groups)):
        train_data = X_train[train_idx, :], y_labels[train_idx]
        valid_data = X_train[valid_idx, :], y_labels[valid_idx]

        _ = pipe.fit(X_train[train_idx, :], y_labels[train_idx])
        preds = pipe.predict(X_train[valid_idx, :])
        auc = roc_auc_score(y_labels[valid_idx], preds)
        aucs.append(auc)

    print(f'Trial done: AUC values on folds: {aucs}')
    return cv_fold_func(aucs)


gc.collect()
np.seterr(over='ignore')
FIT_XGB = True

n_trials = 60

if FIT_XGB:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

best_params = trial.params

best_params['missing'] = -999
best_params['tree_method'] = 'gpu_hist'
clf = xgb.XGBClassifier(**best_params)

pipe_xgb = Pipeline(steps=[
    ('imputer', imp_mean),
    ('scaler', scaler),
    ('xgb', clf)
])

pipe_xgb.fit(
    train_data.query('date > 100')[
        train_data.columns[train_data.columns.str.contains('feature')]
    ].values,
    (train_data.query('date > 100')['resp'] > 0).astype(int).values
)

gc.collect()
pipe_prod = pipe_xgb
env = janestreet.make_env()
env_iter = env.iter_test()

for (test_df, pred_df) in tqdm(env_iter):
    if test_df['weight'].item() > 0:
        pred_df.action = pipe_prod.predict(test_df.loc[:, feature_names].values)
    else:
        pred_df.action = 0
    env.predict(pred_df)

# **Both LightGBM and XGBoost use the GPU.**
# ##### Pipeline:
#
# - Imports
# - Data Loading
# - PurgedGroupTimeSeriesSplitStacking class definition
# - Optuna Parameters optimization
# - Refit with Best Params
# - Submission
name_dict = {True: 'With_Stacking_Set',
             False: 'No_Stacking_Set'}


def plot_cv_indices_stacking(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)  # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, indices_split in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups

        indices = np.array([np.nan] * len(X))
        indices[indices_split[0]] = 1
        indices[indices_split[1]] = 0
        if cv.stacking_mode:
            indices[indices_split[2]] = -1

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    if cv.stacking_mode:
        ax.scatter(range(len(X)), [ii + 3.5] * len(X),
                   c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 2.2, -.2], xlim=[0, len(y)])

    ax.set_title('{}'.format(name_dict[cv.stacking_mode]), fontsize=15)
    # ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


train = (dt.fread(os.path.join(root_path, "train.csv")).to_pandas()
         .query('weight > 0').pipe(reduce_mem_usage)
         .reset_index(drop=True))

train['action'] = (train.resp > 0).astype(int)

resp_cols = [i for i in train.columns if 'resp' in i]

features_names = [i for i in train.columns if 'feature_' in i]
features_index = list(map(lambda x: int(re.sub("feature_", "", x)), features_names))
features_tuples = sorted(list(zip(features_names, features_index)), key=lambda x: x[1])
just_features = [i[0] for i in features_tuples]

N_SPLITS = 5
STACKING_MODE = True
VAL_GROUP_GAP = 20  # Days between end of training set and start of validation set
TEST_GROUP_GAP = 20  # Days between end of validation set and start of testing/stacking set
MAX_DAYS_TRAIN = 120
MAX_DAYS_VAL = 60
MAX_DAYS_TEST = 60
RANDOM_SEED = 28

cv = PurgedGroupTimeSeriesSplitStacking(n_splits=N_SPLITS,
                                        stacking_mode=STACKING_MODE,
                                        max_train_group_size=MAX_DAYS_TRAIN, max_val_group_size=MAX_DAYS_VAL,
                                        max_test_group_size=MAX_DAYS_TEST, val_group_gap=VAL_GROUP_GAP,
                                        test_group_gap=TEST_GROUP_GAP)

# Use the following to test your pipeline
cv_dummy = PurgedGroupTimeSeriesSplitStacking(n_splits=2,
                                              stacking_mode=STACKING_MODE,
                                              max_train_group_size=1, max_val_group_size=1,
                                              max_test_group_size=1, val_group_gap=1,
                                              test_group_gap=1)

for fold, (train_idx, val_idx, test_idx) in enumerate(cv.split(train[just_features], train['action'], train['date'])):
    print("FOLD: {}\n".format(fold))
    print("First train_day: {}\t Last train_day: {} \n".format(train.loc[min(train_idx), 'date'],
                                                               train.loc[max(train_idx), 'date']))
    print("First val_day: {}\t Last val_day: {} \n".format(train.loc[min(val_idx), 'date'],
                                                           train.loc[max(val_idx), 'date']))
    print("First test_day: {}\t Last test_day: {} \n\n\n".format(train.loc[min(test_idx), 'date'],
                                                                 train.loc[max(test_idx), 'date']))

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
plot_cv_indices_stacking(cv, train[just_features], train['action'], train['date'], ax, 5, lw=20)

N_TRIALS = 20


def objective(trial, cv=cv_dummy):
    # xgb parameters
    param_xgb = {
        "xgb_verbosity": 0,
        "xgb_gpu_hist": 1,
        "xgb_objective": "binary:logistic",
        "xgb_booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "xgb_lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "xgb_alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "xgb_max_depth": trial.suggest_int("max_depth", 1, 9),
        "xgb_eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        "xgb_gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "xgb_grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }

    param_lgb = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device": "gpu",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model_1 = XGBClassifier(**param_xgb)
    model_2 = LGBMClassifier(**param_lgb)
    classifier_name = trial.suggest_categorical("classifier", ["DecisionTree", "LogisticRegression"])

    if classifier_name == "DecisionTree":
        final_estimator = DecisionTreeClassifier()
    else:
        final_estimator = LogisticRegression()
    val_aucs = []
    aucs = []
    for kfold, (train_idx, val_idx, test_idx) in enumerate(cv.split(train[just_features].values,
                                                                    train['action'].values,
                                                                    groups=train['date'].values)):
        model_1.fit(train.loc[train_idx, just_features].fillna(-999), train.loc[train_idx, 'action'])
        print('Fitted {}'.format(type(model_1).__name__))
        model_2.fit(train.loc[train_idx, just_features].fillna(-999), train.loc[train_idx, 'action'])
        print('Fitted {}'.format(type(model_2).__name__))
        rf_val_prob = model_1.predict_proba(train.loc[val_idx, just_features].fillna(-999))[:, 1]
        svc_val_prob = model_2.predict_proba(train.loc[val_idx, just_features].fillna(-999))[:, 1]
        val_true = train.loc[val_idx, 'action'].values

        rf_val_pred = np.ones(len(val_idx))
        rf_val_pred[rf_val_prob < 0.5] = 0
        svc_val_pred = np.ones(len(val_idx))
        svc_val_pred[svc_val_prob < 0.5] = 0
        print('{} Val Auc: {}\t {} Val Auc: {}'.format(type(model_1).__name__,
                                                       roc_auc_score(val_true, rf_val_pred),
                                                       type(model_2).__name__,
                                                       roc_auc_score(val_true,
                                                                     svc_val_pred)))

        X_estimator = np.concatenate((np.expand_dims(rf_val_prob, 1), np.expand_dims(svc_val_prob, 1)), 1)

        final_estimator.fit(X_estimator, val_true)
        test_true = train.loc[test_idx, 'action'].values
        rf_test_prob = np.expand_dims(model_1.predict_proba(train.loc[test_idx, just_features].fillna(-999))[:, 1], 1)
        svc_test_prob = np.expand_dims(model_2.predict_proba(train.loc[test_idx, just_features].fillna(-999))[:, 1], 1)
        preds = final_estimator.predict(np.concatenate((rf_test_prob, svc_test_prob), axis=1))

        auc = roc_auc_score(test_true, preds)

        print('Classifier: {}\tFold: {}\t AUC: {}\n'.format(type(final_estimator).__name__, kfold, auc))
        aucs.append(auc)

    print('Average AUC: {}'.format(np.average(auc)))
    return np.average(aucs)


study = optuna.create_study(study_name='stacking_parameter_opt', direction="maximize")
study.optimize(objective, n_trials=2)  # Here use N_TRIALS when doing it properly
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_params = trial.params

model_trained = joblib.load("/kaggle/input/lightgbm-model/lightgbm_model.pickle")

env = janestreet.make_env()
print('Creating submissions file...', end='')
rcount = 0
ts_ids = []
I_WANT_TO_SUBMIT = True
if I_WANT_TO_SUBMIT:
    for (test_df, prediction_df) in env.iter_test():
        X_test = test_df.loc[:, just_features].fillna(-999)
        y_preds = model_trained.predict(X_test.values)
        prediction_df.action = y_preds.item()
        env.predict(prediction_df)
        rcount += len(test_df.index)
    print(f'Finished processing {rcount} rows.')
