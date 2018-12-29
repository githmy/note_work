
# coding: utf-8

# In[1]:


import os
import time
import jieba
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

tagcol = "整理后一级标签"
outtail = "cro_board_flat_only_mean_reg_batch_step"
filtlevel = "美食"
filtlevel = None
insize = 25
outsize = 5


# In[2]:


def read_xls(fname, sheet):
    data = pd.read_excel(io=fname, sheet_name=sheet, header=0)
    return data


# In[3]:


def write_xls(fname, sheet, data):
    data.to_excel(fname, sheet_name=sheet, index=False)


# In[4]:


def bachcut(instr):
    outstr = " ".join(jieba.cut(str(instr).strip()))
    return outstr

def charcut(instr):
    if isinstance(instr,str):
        return " ".join([i1 for i1 in instr])
    else:
        return instr

def dataclean1(data):
    data["label_1_2"] = data["整理后一级标签"] +  ":" + data["整理后二级标签"]
    data["label_1_2"] = data["label_1_2"].replace(" ", "")
    data["店铺名称"] = data["店铺名称"].replace(" ", "")
    return data

def dataclean2(data):
    datao = data[["整理后一级标签", "整理后二级标签", "百度原始TAG", "店铺名称", "label_1_2"]]
#     for i1 in datao.columns:
#         datao[i1] = datao[i1].map(bachcut)
    if filtlevel is not None:
        datao = datao[(datao["整理后一级标签"]==filtlevel)]

    for i1 in ["店铺名称", tagcol]:
        datao[i1] = datao[i1].map(charcut)
    lab = datao.groupby([datao[tagcol]]).count()
    print(lab)

    lab = datao.groupby([datao[tagcol]]).count()

    gid = {}
    gidant = {}
    for i1,i2 in enumerate(lab.index):
        gid[str(i1)]=i2
        gidant[str(i2)]=i1

    #         数据集拆分
    datao = datao[["整理后一级标签", "整理后二级标签", "店铺名称", "label_1_2"]]
    print(datao.shape)
    testdata = datao[(datao["整理后二级标签"].isnull())]
#     testdata = datao[(datao["整理后二级标签"]  == "nan")]

    traindata = datao[(datao["整理后二级标签"].notnull())]
    traindata = traindata.sample(frac=1).reset_index(drop=True)  
    validata = datao[(datao["整理后二级标签"].notnull())]
    lenth_train = traindata.shape[0]
    spint = int(0.8*lenth_train)
    validata = traindata.loc[spint:,:]
    traindata = traindata.loc[0:spint,:]

    return traindata, validata, testdata,gid,gidant


# In[5]:


fname = "bdtag_phone_1108.xlsx"
sheet = "res"
data = read_xls(fname, sheet)


# In[6]:


def get_coefs(word, *arr): return word, np.asarray(arr, dtype=np.float32)

EMBEDDING_FILE = 'wiki.zh.vec'
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf-8"))


# In[7]:


emd_size = len(embeddings_index.get("开"))

datao = dataclean1(data)
traindata, validata, testdata, dict_anti, dict_users = dataclean2(datao)

users_lent = len(dict_users)


# In[8]:


# label_matrix = np.full((users_lent, outsize, emd_size),np.nan)
label_matrix = np.full((users_lent, outsize, emd_size),0.,dtype=np.float32)

# train_matrix = np.full((traindata.shape[0], insize, emd_size), np.nan)
train_matrix = np.full((traindata.shape[0], insize, emd_size), 0.,dtype=np.float32)
train_hots_matrix = np.full((traindata.shape[0], users_lent), np.zeros)

# hot原序列
train_map_label = traindata[tagcol].map(dict_users.get)
labl_hot = np.eye(users_lent)
for d1, i1 in enumerate(train_map_label):
    train_hots_matrix[d1, :] = labl_hot[int(i1), :]

valid_matrix = np.full((validata.shape[0], insize, emd_size), 0.,dtype=np.float32)
# valid_matrix = np.full((validata.shape[0], insize, emd_size), np.nan)
valid_hots_matrix = np.full((validata.shape[0], users_lent), np.zeros)

valid_map_label = validata[tagcol].map(dict_users.get)
for d1, i1 in enumerate(valid_map_label):
    valid_hots_matrix[d1, :] = labl_hot[int(i1), :]

test_matrix = np.full((testdata.shape[0], insize, emd_size), 0.,dtype=np.float32)


# In[9]:


for d1, i1 in enumerate(dict_users.keys()):
    d2 = 0
    for word in str(i1).split():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            label_matrix[d1,d2,:] = embedding_vector
        d2 += 1
        if d2 == outsize:
            break

# 把某列展成向量
def encode_mx(orimat, dataf, colname, bsize):
    c2 = 0
    for d1, i1 in enumerate(dataf[colname]):
        d2 = 0
        for word in str(i1).split():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                orimat[d1,d2,:] = embedding_vector
            else:
               c2 += 1
            d2 += 1
            if d2 == bsize:
                break

encode_mx(train_matrix, traindata, "店铺名称", insize)
encode_mx(valid_matrix, validata, "店铺名称", insize)
encode_mx(test_matrix, testdata, "店铺名称", insize)


# In[ ]:


def gene_model(label_matrix, train_matrix, train_hots_matrix, reg, batch_size, lr_start, lr_min, lr_decay, lr_num):
    # 文件一，建立模型    
    insize = train_matrix.shape[1]
    outsize = label_matrix.shape[1]
    users_lent = train_hots_matrix.shape[1]
    emb_size = label_matrix.shape[2]
#     inputs_p = tf.placeholder(tf.float32, name='inputs', shape=[None, insize, emb_size])
#     labels_p = tf.placeholder(tf.float32, name='labels', shape=[None, users_lent])
    inputs_p = tf.placeholder(tf.float32, name='inputs', shape=[insize, emb_size])
    labels_p = tf.placeholder(tf.float32, name='labels', shape=[users_lent])

    # 第一层 常数
    useritemvec = []  # usern
    for i1 in range(users_lent):
        # 每个输入的最近距离
        item_outword_vec = []
        for i2 in range(outsize):
            # 每个out 输出 的最近距离的标量
            sub_new = tf.subtract(inputs_p,
                                  tf.tile(tf.constant(label_matrix[i1, i2:i2 + 1, :], tf.float32), [insize, 1]))
            #             item_outword_vec.append(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(sub_new), 1))))
            #             item_outword_vec.append(tf.reduce_min(tf.reduce_mean(sub_new, 1)))
            item_outword_vec.append(tf.reduce_mean(sub_new, 1))
        # item_outword_vec.append(tf.reduce_min(tf.reduce_mean(sub_new, 1)/emb_size))
        useritemvec.append(tf.stack(item_outword_vec, axis=0))
    midmatr = tf.reshape(tf.stack(useritemvec, axis=0), [-1, 1])

    myreg1 = layers.l1_regularizer(reg)  # 创建一个正则化方法， 0.01为系数，相当于给每个参数前乘以0.01,当然这里也可以是l2方法或者sum混合方法
    with tf.variable_scope('var', initializer=tf.random_normal_initializer(),
                           regularizer=myreg1):  # 高能！：参数里面指明了regularizer
        Weights = tf.get_variable('W', shape=[users_lent, users_lent * outsize * insize],
                                  initializer=tf.random_normal_initializer())  # 逻辑函数
#     Weights = tf.Variable(tf.random_normal([users_lent, users_lent * outsize * insize], name='W'))
    biases = tf.Variable(tf.random_normal([users_lent, 1], name='b'))
    mult = tf.matmul(Weights, midmatr)
    #     Wx_plus_b = tf.add(tf.matmul(Weights, midmatr), biases)
    #     Wx_plus_b = tf.add(mult, biases)
    #     labels_p_ = tf.sigmoid(Wx_plus_b)
    labels_p_ = tf.reshape(tf.add(mult, biases), [-1], name="predicts")
    tf.summary.histogram('predicts', labels_p_)  # 记录向量或者矩阵，tensor的数值分布变化。

    losss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=labels_p_, labels=labels_p, dim=-1),
                           name="losses")
    tf.summary.scalar('losses', losss)  # 记录标量的变化

    tmplr = tf.Variable(tf.constant(lr_start), name="lr")
    uplr = tf.assign(tmplr, tf.multiply(tf.constant(lr_decay), tmplr))
    tf.summary.scalar('lr', tmplr)  # 记录标量的变化

    train_op = tf.train.AdamOptimizer(tmplr).minimize(losss)
    # 设置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs_%s/" % outtail, sess.graph)
        sess.run(tf.global_variables_initializer())
        epochn = 1000
        samn = train_matrix.shape[0]
        tmplr = lr_start
        print("start training")
        for d1, epoch1 in enumerate(range(epochn)):
            startt = time.time()
#             b_start = 0
#             b_end = b_start + batch_size
            for step in range(samn):
#             step = 0
#             while b_end < samn:
                feed_dict = {
                    inputs_p: train_matrix[step, :, :],
                    labels_p: train_hots_matrix[step, :],
                }
                # 更新速度
                if step % lr_num == 0 and tmplr > lr_min:
                    tmplr = sess.run(uplr)
                losp, _ = sess.run([losss, train_op], feed_dict)
                if step % 10000 == 0:
                    result, vecb = sess.run([merged, labels_p_], feed_dict)
                    writer.add_summary(result, step + d1 * samn)
                    print("step: %s/%s. loss is: %s" % (step, samn, losp))
                    save_path = saver.save(sess, 'model_%s/filename.ckpt' % outtail, global_step=step + d1 * samn)
#                 b_start += batch_size
#                 b_end = b_start + batch_size
                step += 1
            print("epoch: %s/%s. time: %s s" % (epoch1, epochn, time.time() - startt))
        writer.close()
    return 0

reg = 0.01
batch_size = 2
lr_start = 0.001
lr_min = 1e-11
lr_decay = 0.998
lr_num = 1000
gene_model(label_matrix, train_matrix, train_hots_matrix, reg,batch_size,lr_start,lr_min,lr_decay,lr_num)


# In[ ]:


reg = 0.01
batch_size = 2
lr_start = 0.001
lr_min = 1e-11
lr_decay = 0.998
lr_num = 1000
gene_model(label_matrix, train_matrix, train_hots_matrix, reg,batch_size,lr_start,lr_min,lr_decay,lr_num)


# In[ ]:


def predic(valid_matrix):
    samn = valid_matrix.shape[0]
    latest_ckpt = tf.train.latest_checkpoint("model_%s" % outtail)
    print(latest_ckpt)
    saver = tf.train.import_meta_graph(latest_ckpt+".meta")
#     saver = tf.train.import_meta_graph('model_cro/filename.ckpt-5800.meta')
    with tf.Session() as sess:
        saver.restore(tf.get_default_session(), latest_ckpt)
        tf.tables_initializer()
        x_test_batch = []
        for d2,i2 in enumerate(range(samn)):
            x_test_batch.append(sess.run("predicts:0", {"inputs:0": valid_matrix[d2,:,:]}))
    return x_test_batch

# outtail="rel"
val_resmat = predic(valid_matrix)
val_resmat = np.array(val_resmat)


# In[ ]:


# 评估
val_lst = np.argmax(val_resmat, axis=1)
val_lent = len(val_lst)
count = 0
for d1, i1 in enumerate(valid_map_label):
    if int(i1) == val_lst[d1]:
        count += 1
print("accuracy is: %s in %s classes." % (count/val_lent,users_lent))


# In[ ]:


# 结果
tes_resmat = predic(test_matrix)
tes_resmat = np.array(tes_resmat)
tes_lst = np.argmax(tes_resmat, axis=1)
tes_lent = len(tes_lst)
over_list = []
for i1 in tes_lst:
    over_list.append(dict_anti.get(str(i1)))
over_pd=datao[(datao["整理后二级标签"].isnull())]
over_pd[tagcol]=over_list


# In[ ]:



write_xls("%s_%s" % (outtail, fname), sheet, over_pd)

