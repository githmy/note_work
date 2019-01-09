# -*- coding: utf-8 -*-
# 某个项目的数据和模型
import pandas as pd
import os
import re

datapath = './daily'
filepath = os.path.join(datapath, os.listdir('./daily')[0])

ticker_regex = re.compile('.+_(?P<ticker>.+)\.csv')
get_ticker = lambda x: ticker_regex.match(x).groupdict()['ticker']
print(filepath, get_ticker(filepath))

ret = lambda x, y: log(y / x)  # Log return
zscore = lambda x: (x - x.mean()) / x.std()  # zscore

D = pd.read_csv(filepath, header=None, names=['UNK', 'o', 'h', 'l', 'c', 'v'])  # Load the dataframe with headers
D.head()


# UNK	o	h	l	c	v
# 20070702	0	34.1445	35.3901	33.9529	34.8065	1834373.873
# 20070703	0	34.8239	35.3726	34.0400	34.5452	4143385.114
# 20070705	0	34.4929	34.8413	34.1793	34.5365	4521413.667
# 20070706	0	34.3187	34.7716	33.7961	34.6410	4021547.844
# 20070709	0	34.4493	35.1462	33.5173	35.0329	5388421.001

def make_inputs(filepath):
    D = pd.read_csv(filepath, header=None, names=['UNK', 'o', 'h', 'l', 'c', 'v'])  # Load the dataframe with headers
    D.index = pd.to_datetime(D.index, format='%Y%m%d')  # Set the indix to a datetime
    Res = pd.DataFrame()
    ticker = get_ticker(filepath)

    Res['c_2_o'] = zscore(ret(D.o, D.c))
    Res['h_2_o'] = zscore(ret(D.o, D.h))
    Res['l_2_o'] = zscore(ret(D.o, D.l))
    Res['c_2_h'] = zscore(ret(D.h, D.c))
    Res['h_2_l'] = zscore(ret(D.h, D.l))
    Res['c1_c0'] = ret(D.c, D.c.shift(-1)).fillna(0)  # Tommorows return
    Res['vol'] = zscore(D.v)
    Res['ticker'] = ticker
    return Res


Res = make_inputs(filepath)

Res.head()  # Lets look at what we got
# c_2_o	h_2_o	l_2_o	c_2_h	h_2_l	c1_c0	vol

Res.corr()  # Quick check to see we didn't mess it up. All
# c_2_o	h_2_o	l_2_o	c_2_h	h_2_l	c1_c0	vol
# c_2_o	1.000000	0.727980	0.646456	0.664281	-0.076033	0.008279	0.011871
# h_2_o	0.727980	1.000000	0.157799	-0.028889	-0.659993	0.007359	0.300114
# l_2_o	0.646456	0.157799	1.000000	0.770473	0.637713	0.010860	-0.321847
# c_2_h	0.664281	-0.028889	0.770473	1.000000	0.608713	0.004047	-0.309896
# h_2_l	-0.076033	-0.659993	0.637713	0.608713	1.000000	0.002522	-0.478965
# c1_c0	0.008279	0.007359	0.010860	0.004047	0.002522	1.000000	-0.017168
# vol	0.011871	0.300114	-0.321847	-0.309896	-0.478965	-0.017168	1.000000

# Generating the full data set
Final = pd.DataFrame()
for f in os.listdir(datapath):
    filepath = os.path.join(datapath, f)
    if filepath.endswith('.csv'):
        Res = make_inputs(filepath)
        Final = Final.append(Res)

Final.head()

pivot_columns = Final.columns[:-1]
P = Final.pivot_table(index=Final.index, columns='ticker', values=pivot_columns)  # Make a pivot table from the data

P.head()

mi = P.columns.tolist()

new_ind = pd.Index(e[1] + '_' + e[0] for e in mi)

P.columns = new_ind
P = P.sort(axis=1)  # Sort by columns

P.head()

clean_and_flat = P.dropna(1)

target_cols = list(filter(lambda x: 'c1_c0' in x, clean_and_flat.columns.values))
input_cols = list(filter(lambda x: 'c1_c0' not in x, clean_and_flat.columns.values))

InputDF = clean_and_flat[input_cols][:3900]
TargetDF = clean_and_flat[target_cols][:3900]

corrs = TargetDF.corr()

num_stocks = len(TargetDF.columns)

# If i put one dollar in each stock at the close, this is how much I'd get back
TotalReturn = ((1 - exp(TargetDF)).sum(1)) / num_stocks


def labeler(x):
    if x > 0.0029:
        return 1
    if x < -0.00462:
        return -1
    else:
        return 0


Labeled = pd.DataFrame()
Labeled['return'] = TotalReturn
Labeled['class'] = TotalReturn.apply(labeler, 1)
Labeled['multi_class'] = pd.qcut(TotalReturn, 11, labels=range(11))

pd.qcut(TotalReturn, 5).unique()


def labeler_multi(x):
    if x > 0.0029:
        return 1
    if x < -0.00462:
        return -1
    else:
        return 0


Labeled['class'].value_counts()

Labeled['act_return'] = Labeled['class'] * Labeled['return']

Labeled[['return', 'act_return']].cumsum().plot(subplots=True)

# Making a baseline
from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1e5)

test_size = 600

res = logreg.fit(InputDF[:-test_size], Labeled['multi_class'][:-test_size])

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(Labeled['multi_class'][-test_size:], res.predict(InputDF[-test_size:])))
print(confusion_matrix(Labeled['multi_class'][-test_size:], res.predict(InputDF[-test_size:])))

Labeled['predicted_action'] = list(map(lambda x: -1 if x < 5 else 0 if x == 5 else 1, res.predict(InputDF)))
print(confusion_matrix(Labeled['class'][-test_size:], Labeled['predicted_action'][-test_size:]))

Labeled['pred_return'] = Labeled['predicted_action'] * Labeled['return']

Res = Labeled[-test_size:][['return', 'act_return', 'pred_return']].cumsum()
Res[0] = 0
Res.plot()

# Training a basic feed forward network
import tensorflow as tf
from  tensorflow.contrib.learn.python.learn.estimators.dnn import DNNClassifier
from tensorflow.contrib.layers import real_valued_column

Labeled['tf_class'] = Labeled['multi_class']
num_features = len(InputDF.columns)
dropout = 0.2
hidden_1_size = 1000
hidden_2_size = 250
num_classes = Labeled.tf_class.nunique()
NUM_EPOCHS = 100
BATCH_SIZE = 50
lr = 0.0001

train = (InputDF[:-test_size].values, Labeled.tf_class[:-test_size].values)
val = (InputDF[-test_size:].values, Labeled.tf_class[-test_size:].values)
NUM_TRAIN_BATCHES = int(len(train[0]) / BATCH_SIZE)
NUM_VAL_BATCHES = int(len(val[1]) / BATCH_SIZE)

len(InputDF)


class Model():
    def __init__(self):
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
        self.target_data = tf.placeholder(dtype=tf.int32, shape=[None])
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=[])
        with tf.variable_scope("ff"):
            droped_input = tf.nn.dropout(self.input_data, keep_prob=self.dropout_prob)

            layer_1 = tf.contrib.layers.fully_connected(
                num_outputs=hidden_1_size,
                inputs=droped_input,
            )
            layer_2 = tf.contrib.layers.fully_connected(
                num_outputs=hidden_2_size,
                inputs=layer_1,
            )
            self.logits = tf.contrib.layers.fully_connected(
                num_outputs=num_classes,
                activation_fn=None,
                inputs=layer_2,
            )
        with tf.variable_scope("loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target_data)
            mask = (1 - tf.sign(1 - self.target_data))  # Don't give credit for flat days
            mask = tf.cast(mask, tf.float32)
            self.loss = tf.reduce_sum(self.losses)

        with tf.name_scope("train"):
            opt = tf.train.AdamOptimizer(lr)
            gvs = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(gvs, global_step=global_step)

        with tf.name_scope("predictions"):
            self.probs = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.probs, 1)
            correct_pred = tf.cast(tf.equal(self.predictions, tf.cast(self.target_data, tf.int64)), tf.float64)
            self.accuracy = tf.reduce_mean(correct_pred)


with tf.Graph().as_default():
    model = Model()
    input_ = train[0]
    target = train[1]
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run([init])
        epoch_loss = 0
        for e in range(NUM_EPOCHS):
            if epoch_loss > 0 and epoch_loss < 1:
                break
            epoch_loss = 0
            for batch in range(0, NUM_TRAIN_BATCHES):
                start = batch * BATCH_SIZE
                end = start + BATCH_SIZE
                feed = {
                    model.input_data: input_[start:end],
                    model.target_data: target[start:end],
                    model.dropout_prob: 0.9
                }
                _, loss, acc = sess.run(
                    [
                        model.train_op,
                        model.loss,
                        model.accuracy,
                    ]
                    , feed_dict=feed
                )
                epoch_loss += loss
            print('step - {0} loss - {1} acc - {2}'.format((1 + batch + NUM_TRAIN_BATCHES * e), epoch_loss, acc))



def main():
    pass


if __name__ == '__main__':
    # 1. 测试
    main()
