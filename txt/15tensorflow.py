from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


# 命令行参数接收
def argv_paras():
    # python tt.py --str_name test_str --int_name 99 --bool_name True
    flags = tf.flags
    FLAGS = flags.FLAGS
    FLAGS.DEFINE_string('str_name', 'def_v_1', "descrip1")
    FLAGS.DEFINE_integer('int_name', 10, "descript2")
    FLAGS.DEFINE_boolean('bool_name', False, "descript3")
    # 定义必要的参数
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    print(FLAGS.str_name)
    print(FLAGS.int_name)
    print(FLAGS.bool_name)
    tf.app.run()


# 动态加载
def eager_exection():
    import tensorflow as tf
    tf.enable_eager_execution()
    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)
    print(C)

    x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))
    # 梯度
    with tf.GradientTape() as tape:  # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
        y = tf.square(x)
    y_grad = tape.gradient(y, [x])  # 计算 y 关于 x 的导数
    print([y.numpy(), y_grad.numpy()])
    # [array([9.], dtype=float32), array([6.], dtype=float32)]


# 图像维度数顺序   [batch, height, width, channels]
#                [batch_size, num_steps, emb_size]
# feed data 形式
def feed_format():
    ordinary_data = [[数据行1, 数据行2, 数据行3], [数据标1, 数据标2, 数据标3]]
    sequence_data = [[[数据行1字1, 数据行1字2, 数据行1字3], [数据行2字1, 数据行2字2, 数据行2字3]],
                     [[数据标1字1, 数据标1字2, 数据标1字3], [数据标2字1, 数据标2字2, 数据标2字3]]]
    _, chars, segs, tags = sequence_data
    feed_dict = {
        # 一个batch的标量数据，为一个数组
        char_inputs: np.asarray(chars),
        seg_inputs: np.asarray(segs),
        dropout: 1.0,
    }
    if is_train:
        feed_dict[targets] = np.asarray(tags)
        feed_dict[dropout] = config["dropout_keep"]
    return feed_dict


# 数据操作
def data_manipulate():
    # # 1. 将dataset缓存在内存或者本地硬盘,默认是内存
    # cache(filename='')
    # # 2. 将 [file, label] 变换到 [img_data, label],预处理一般用cpu
    # map(
    #     map_func,
    #     num_parallel_calls=None
    # )
    def _mapfunc(file, label):
        with tf.device('/cpu:0'):
            img_raw = tf.read_file(file)
            decoded = tf.image.decode_bmp(img_raw)
            resized = tf.image.resize_images(decoded, [h, w])
        return resized, label

    # # 3. shuffle
    # shuffle(
    #     buffer_size,
    #     seed=None,
    #     reshuffle_each_iteration=None
    # )
    # reshuffle_each_iteration 默认 True
    # buffer_size 比样本数+1
    # 9. 使用组合
    # 0. 获取目录标签
    filelist = os.listdir(img_dir)
    # lable_list = ... # 标签列表根据自己的情况获取
    # 两个tensor
    t_flist = tf.constant(filelist)
    t_labellist = tf.constant(lable_list)
    # 构造 Dataset
    dataset = tf.data.Dataset().from_tensor_slices(t_flist, t_labellist)
    # 2. map文件
    dset = dataset.map(_mapfunc)
    _iter = dset.make_one_shot_iterator()
    next_one = _iter.get_next()
    # next_one 作为tensor正常使用. type: tuple
    img, label = next_one
    # 一般会报 channel 最后这个维度为 None
    # 必须加这个reshape
    out = tf.reshape(img, [-1, h, w, c])


# 加载数据文件
def read_csv(line):
    FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                      [0.0], [0]]
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    label = fields[-1]
    label = tf.cast(label, tf.int64)
    features = tf.stack(fields[0:-1])
    return features, label


def read_any():
    log_file = os.path.join("out_dir", "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len([1, 3, 5]))
    return 0


def read_txt():
    src_dataset = tf.data.TextLineDataset('src_data.txt')
    tgt_dataset = tf.data.TextLineDataset('tgt_data.txt')
    return 0


# tf格式 写入文件
def writedata():
    import tensorflow as tf
    xlist = [[1, 2, 3], [4, 5, 6, 8]]
    ylist = [1, 2]
    # 这里的数据只是举个例子来说明样本的文本长度不一样，第一个样本3个词标签1，第二个样本4个词标签2
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for i in range(2):
        x = xlist[i]
        y = ylist[i]
        example = tf.train.Example(features=tf.train.Features(feature={
            "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
            'x': tf.train.Feature(int64_list=tf.train.BytesList(value=x))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read2buckets():
    feature_names = ['x']

    def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
        def parse(example_proto):
            features = {"x": tf.VarLenFeature(tf.int64),
                        "y": tf.FixedLenFeature([1], tf.int64)}
            parsed_features = tf.parse_single_example(example_proto, features)
            x = tf.sparse_tensor_to_dense(parsed_features["x"])
            x = tf.cast(x, tf.int32)
            x = dict(zip(feature_names, [x]))
            y = tf.cast(parsed_features["y"], tf.int32)
            return x, y

        # 1. io 异步
        dataset = (tf.contrib.data.TFRecordDataset(file_path).map(parse))
        files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
        dataset = files.interleave(tf.data.TFRecordDataset)  # 现在该函数已经加了cycle_length参数
        dataset = files.apply(
            tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
        # 2. 数据 预加工 异步
        # dataset = dataset.map(map_func=parse_fn, num_parallel_calls=FLAGS.num_parallel_calls)
        # dataset = dataset.batch(batch_size=FLAGS.batch_size)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=FLAGS.batch_size))
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.repeat(repeat_count)
        # tf.contrib.data.shuffle_and_repeat(repeat_count)
        dataset = dataset.padded_batch(2, padded_shapes=({'x': [6]}, [1]))  # batch size为2，并且x按maxlen=6来做padding
        # 3. 训练和数据加载异步
        dataset = dataset.batch(batch_size=FLAGS.batch_size)
        dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)  # last transformation
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels

    next_batch = my_input_fn('train.tfrecords', True)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1):
            xs, y = sess.run(next_batch)
            print(xs['x'])
            print(y)


def dim_reverse():
    # 第0维 序列内容翻转
    outputs_bw = tf.reverse(outputs_bw, [0])
    # 第2维 序列内容拼接
    output = tf.concat([outputs_fw, outputs_bw], 2)
    # t1 = [[1, 2, 3], [4, 5, 6]]
    # t2 = [[7, 8, 9], [10, 11, 12]]
    # tf.concat(0, [t1, t2]) == > [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    # tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    # 第2维 维度交换 [0,1,2]
    output = tf.transpose(output, perm=[1, 0, 2])
    # 第2维 序列reshape 维度, 变成一维时，先从最内一维展开，逐渐推到前维。反之先从最后一维填充，逐渐推到前维。
    output = tf.reshape(output, [-1, 2, 5, 4])
    """
    [[[[1 2 3 4]
       [5 6 7 8]
       [7 6 5 4]
       [3 2 1 0]
       [3 3 3 3]]    
      [[3 3 3 3]
       [1 1 1 1]
       [1 1 1 1]
       [2 2 2 2]
       [2 2 2 2]]]]
    """
    # 维度 张量缩并
    # v (B,T,A)，u_omega = tf.Variable(tf.random_normal([A], stddev=0.1))
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    # 矩阵的合并和分解
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.stack([a, b], axis=0)
    d = tf.unstack(c, axis=0)
    e = tf.unstack(c, axis=1)
    print(c.get_shape())  # 获取维度。
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c))
        print(sess.run(d))
        print(sess.run(e))
    """
    (2, 3)
    [[1 2 3] 
     [4 5 6]] 
    [array([1, 2, 3]), array([4, 5, 6])] 
    [array([1, 4]), array([2, 5]), array([3, 6])]
    """
    # 去掉维度为1的维度
    #  't' 是一个维度是[1, 2, 1, 3, 1, 1]的张量
    tf.shape(tf.squeeze(t))  # [2, 3]， 默认删除所有为1的维度
    # 't' 是一个维度[1, 2, 1, 3, 1, 1]的张量
    tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]，标号从零开始，只删掉了2和4维的1
    # 维度切分
    a = np.reshape(range(24), (4, 2, 3))
    """
    [[[ 0,  1,  2],
        [ 3,  4,  5]],
       [[ 6,  7,  8],
        [ 9, 10, 11]],
       [[12, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23]]]
    """
    # 将a分为两个tensor，a.shape(1)为2，可以整除，不会报错。
    # 输出应该为2个shape为[4,1,3]的tensor
    b = tf.split(a, 2, 1)
    """
    [[[ 0,  1,  2]],
       [[ 6,  7,  8]],
       [[12, 13, 14]],
       [[18, 19, 20]]]), array([[[ 3,  4,  5]],
       [[ 9, 10, 11]],
       [[15, 16, 17]],
       [[21, 22, 23]]]
    """
    c = tf.split(a, 2, 0)
    # a.shape(0)为4，被2整除，输出2个[2,2,3]的Tensor

    # 't' is a tensor of shape [2]
    tf.shape(tf.expand_dims(t, 0))
    # ==> [1, 2]
    tf.shape(tf.expand_dims(t, 1))
    # ==> [2, 1]
    # mask 默认最长mask长度，除非指定 如5，掩模 至少1维。
    tf.sequence_mask([1, 3, 2], 5)
    # [[ True False False]
    # [ True  True  True]
    # [ True  True False]]
    tf.sequence_mask([[1, 3], [2, 0]])
    # [[[True, False, False],
    #   [True, True, True]],
    #  [[True, True, False],
    #   [False, False, False]]]
    # 判断维度是否相同
    inputs.shape.assert_is_compatible_with(memory_mask.shape)


# 类型转化
def type_trans():
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t


# 数组操作
def array_var_manipulation():
    # 用常数初始化，
    value = [1, 2, 3, 4]
    tf.constant_initializer(value)
    # 变量初始化，2种方法
    init = tf.initialize_all_variables()
    init_new_vars_op = tf.initialize_variables([v_6, v_7, v_8])
    sess.run(init_new_vars_op)
    # 类型转换
    tf.cast(x, tf.int32)
    inputs = [[1, 0, 2], [3, 2, 4]]
    inputs = np.array(inputs)
    # 符号判断 >0 =1 ；<0 =-1；=0 =0.
    A = tf.sign(inputs)
    # 维度求和，keep_dims维度不变，维度内求和；reduction_indices 缩并的维度
    # 处理的对象为list
    B = tf.reduce_sum(A, reduction_indices=1, keep_dims=False)
    # 遍历匹配最后一维
    tf.nn.bias_add(a, b)
    # 处理的对象为list
    B = tf.add_n(sum_list)
    # 下面方式等价
    output = tf.add_n([input1, input2])
    sess.run(input1 + input2)
    sess.run(output)
    with tf.Session() as sess:
        print(sess.run(A))
        print(sess.run(B))
    # 逻辑与或非
    tf.logical_not(False)
    tf.logical_and(True, False)
    tf.logical_or(True, False)
    # 异或
    tf.bitwise.bitwise_or(predictions_m, input_y_m)
    """
    [[1 0 1],[1 1 1]]
    [2 3]
    """
    # 绝对值
    x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
    tf.abs(x)
    # 最大化低值
    a = [1, 5, 3]
    f1 = tf.maximum(a, 3)
    # [3 5 3]
    # 最小化高值
    f2 = tf.minimum(a, 3)
    # [1 3 3]
    # 最大位置
    f3 = tf.argmax(a, 0)
    # 1
    # 最小位置
    f4 = tf.argmin(a, 0)
    # 0
    # 幂
    tf.pow(a, 2)
    # 数值乘
    w1 = tf.Variable(2.0)
    a = tf.multiply(w1, 3.0)  # 矩阵的维度不同，会自动复制。
    # 矩阵乘
    w1 = tf.tf.matmul(a, b)  # 第一个矩阵的列必须等于第二个矩阵的行，线性矩阵相乘。高维情况，只对最后两维做矩阵乘法。
    # 判断是否相等
    print(sess.run(tf.equal(A, B)))
    # [[True  True  True False False]]
    # 数据变成张量
    A = list([1, 2, 3])
    tf.convert_to_tensor(A)
    # 计算梯度
    tf.gradients(ys, xs)  # 要注意的是，xs中的x必须要与ys相关

    # 节点被 stop之后，这个节点上的梯度，就无法再向前BP了
    c = tf.add(a, b)
    tf.stop_gradient(c)
    grad = tf.gradients(ys=b, xs=a)  # 一阶导
    print(grad[0])
    grad_2 = tf.gradients(ys=grad[0], xs=a)  # 二阶导
    grad_3 = tf.gradients(ys=grad_2[0], xs=a)  # 三阶导


# GPU操作
def gpu_setting():
    init = tf.global_variables_initializer()
    # 设置tensorflow对GPU的使用按需分配
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 2.启动图 (graph)
    sess = tf.Session(config=config)
    # sess = tf.InteractiveSession(config=config)
    sess.run(init)


# 会话的两种方式
def session_2way():
    # 1.
    with tf.Session() as sess:
        result = sess.run([product])
        print(result)
    # 2.
    sess = tf.InteractiveSession()
    x = tf.Variable([1., 2.])
    y = tf.constant([3., 4.])
    # 使用初始化器initializer op的run()方法初始化x
    x.initializer.run()
    # 增加一个减法sub op，从x减去y
    sub = tf.subtract(x, y)
    print(sub.eval())
    # 3.
    target_session = ""
    train_model = model_helper.create_train_model(model_creator, hparams, scope)
    config_proto = utils.get_config_proto(
        log_device_placement=log_device_placement,
        num_intra_threads=hparams.num_intra_threads,
        num_inter_threads=hparams.num_inter_threads)
    train_sess = tf.Session(target=target_session, config=config_proto, graph=train_model.graph)


# 变量初始化方式
def init_func():
    ##正太分布
    tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None, name=None)
    ##正态分布
    tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None, name=None)
    ##平均分布tf.random_uniform(shape,minval=0,maxval=None,dtype=dtypes.float32,seed=None,name=None)
    ##gamma分布
    tf.random_gamma()

    b1 = tf.Variable(tf.zeros(shape=[3]))
    b2 = tf.Variable(tf.ones([2, 2]))
    b3 = tf.Variable(tf.fill([2, 3], 3))
    b4 = tf.Variable(tf.constant(2, dtype=tf.float32, shape=[1, 2]))

    tf.constant_initializer()
    tf.random_normai_initializer()
    tf.truncated_naomal_initializer()
    tf.random_uniformal_initializer()

    # 值域范围 tf.clip_by_value 小于min的让它等于min，大于max的元素的值等于max
    A = np.array([[1, 1, 2, 4], [3, 4, 8, 5]])
    sess.run(tf.clip_by_value(A, 2, 5))
    """
    [[2 2 2 4]
     [3 4 5 5]]
    """


# 变量查看
def check_var_func():
    # 可训练变量的查询
    vs = tf.trainable_variables()
    print(len(vs))
    for v in vs:
        print(v)

    # 所有变量的查询
    a = tf.all_variables()
    print(set(a))

    a = tf.global_variables()
    print(set(a))

    # 其他查看方法
    char_lookup = tf.get_variable(
        name="char_embedding",
        shape=[self.num_chars, self.char_dim],
        initializer=self.initializer)
    char_lookup.read_value()
    char_lookup.assign(emb_weights)


# 可训练方式
def trainable_func():
    # 1. trainable设置为True
    # trainable设置为True，就会把变量添加到GraphKeys.TRAINABLE_VARIABLES集合中，
    # 如果是False，则不添加。
    # 而在计算梯度进行后向传播时，我们一般会使用一个optimizer，然后调用该optimizer的compute_gradients方法。
    # 在compute_gradients中，第二个参数var_list如果不传入，则默认为GraphKeys.TRAINABLE_VARIABLES。
    # 自定义变量列表，那么即使设置了trainable=False，只要把该变量加入到自定义变量列表中，变量还是会参与后向传播的
    # 2. 方式2
    tf.stop_gradient(var)
    # 3. 方式3
    trainable_vars = tf.trainable_variables()
    freeze_conv_var_list = [t for t in trainable_vars if not t.name.startswith(u'conv')]
    grads = opt.compute_gradients(loss, var_list=freeze_conv_var_list)


# 参数更新
def vari_change():
    # tf.assign(ref, value) 的方式来把 value 值赋给 ref 变量
    v3 = tf.Variable(666, name='v3', dtype=tf.int32)
    state = tf.Variable(0, name="counter")
    # 创建一个 op , 其作用是时 state 增加 1
    one = tf.constant(1)  # 直接用 1 也就行了
    new_value = tf.add(state, 1)
    update = tf.assign(state, new_value)
    # 启动图之后， 运行 update op
    with tf.Session() as sess:
        # 创建好图之后，变量必须经过‘初始化’
        sess.run(tf.global_variables_initializer())
        # 查看state的初始化值
        print(sess.run(state))
        for _ in range(3):
            sess.run(update)  # 这样子每一次运行state 都还是1
            print(sess.run(state))


# 梯度更新 静图
def gradients_update():
    opt = tf.train.AdamOptimizer(lr)
    opt = GradientDescentOptimizer(learning_rate=0.1)
    # 函数先计算梯度 for a list of variables
    grads_and_vars = opt.compute_gradients(loss, "< list of variables >")
    # 函数来更新该梯度所对应的参数的状态。
    # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    # need to the 'gradient' part, for example cap them, etc.
    capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

    train_op = opt.apply_gradients(capped_grads_and_vars)
    sess.run(train_op, feed_dict=feed_dict)


#

# 梯度更新 动图
def gradients_update1():
    tf.enable_eager_execution()
    X = tf.constant(X)
    y = tf.constant(y)
    a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
    b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
    variables = [a, b]
    num_epoch = 10000
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    for e in range(num_epoch):
        # 使用 tf.GradientTape() 记录损失函数的梯度信息
        with tf.GradientTape() as tape:
            y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
        # TensorFlow 自动计算损失函数关于自变量（模型参数）的梯度
        grads = tape.gradient(loss, variables)
        # TensorFlow 自动根据梯度更新参数
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))


# 梯度更新
def gradients_update2():
    global_step = tf.Variable(0)
    # 通过exponential_decay函数生成学习率
    learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
    # 使用指数衰减的学习率，在minimize函数中传入global_step将自动更新
    learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 上面这段代码中设定了初始学习率为0.1，因为指定了staircase = True，所以每训练100轮后学习率乘以0.96。


# 梯度限制
def gradients_limit():
    # 方式一
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
    grads = optimizer.compute_gradients(loss)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 5), v)  # 阈值这里设为5
    train_op = optimizer.apply_gradients(grads)
    # 方式二
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads, global_norm = tf.clip_by_global_norm(grads, 5)
    train_op = optimizer.apply_gradients(zip(grads, variables))


# 激活函数
def activate_func():
    y = []
    cur_layer = tf.nn.relu(y)
    cur_layer = tf.nn.relu6(y)
    cur_layer = tf.nn.leaky_relu(y)
    cur_layer = tf.nn.softmax(y)
    Y = tf.placeholder("float", [None, 10])
    ww, wo = [], []
    py_x = tf.matmul(ww, wo)
    # 每一维多个1
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=py_x, labels=Y))
    # 每一维只有一个1
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    # 输入非onehot
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  # construct an optimizer
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    cur_layer = tf.sigmoid(y)
    cur_layer = tf.tanh()
    cur_layer = tf.tanh()

    # 将一个张量中的数值限制在一个范围之内
    cur_layer = tf.clip_by_value(y, 1e-10, 1.0)
    cur_layer = tf.nn.batch_normalization(inputs=y, decay=0.9, updates_collections=None, is_training=False)
    cur_layer = tf.nn.batch_norm_with_global_normalization(inputs=y, decay=0.9, updates_collections=None,
                                                           is_training=False)
    cur_layer = tf.nn.fused_batch_norm(inputs=y, decay=0.9, updates_collections=None, is_training=False)

    weights = []
    tf.contrib.layers.l1_regularizer(.5)(weights)
    tf.contrib.layers.l2_regularizer(.5)(weights)


# drop
def drop_func():
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # inputs*Weight+biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, 0.5)  # 丢弃50%结果，减少过拟合


# 正则
def normal_func():
    # dim 0 按列 正则。# dim 1 按行 正则。
    tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)


def regular_func():
    from tensorflow.contrib import layers

    myreg1 = layers.l1_regularizer(0.01)
    # 创建一个正则化方法， 0.01为系数，相当于给每个参数前乘以0.01,当然这里也可以是l2方法或者sum混合方法
    # weight = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
    # myreg1(weight) 输出为(1²+(-2)²+(-3)²+4²)/2*0.01=0.15 该项会加到weight上
    with tf.variable_scope('var', initializer=tf.random_normal_initializer(),
                           regularizer=myreg1):  # 高能！：参数里面指明了regularizer
        weight = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())  # 逻辑函数
    # get_collection 获得list, reduce_sum进行对list求和
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    with sess.as_default():
        result = regularization_loss.eval()
    print(result)

    # 或
    class mynet:
        def __init__(self):
            self.myreg1 = layers.l1_regularizer(0.01)
            self.inference()

        def inference(self):
            with tf.variable_scope('var', initializer=tf.random_normal_initializer(), regularizer=self.myreg1):
                weight = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())

    # 或 损失正则项 取 L2 范数的值的一半，具体如下： output = sum(t ** 2) / 2
    l2_reg_lambda = 0.01
    l2_reg_loss = tf.constant(0.)
    for var in tf.trainable_variables():
        l2_reg_loss += tf.nn.l2_loss(var)
    loss5 = tf.reduce_mean(-log_likelihood) + l2_reg_lambda * l2_reg_loss

    # 总述
    from tensorflow import contrib
    weight = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
    with tf.Session() as sess:
        # 输出为(|1|+|-2|+|-3|+|4|)*0.5=5
        print(sess.run(contrib.layers.l1_regularizer(0.5)(weight)))
        # 输出为(1²+(-2)²+(-3)²+4²)/2*0.5=7.5
        # TensorFlow会将L2的正则化损失值除以2使得求导得到的结果更加简洁
        print(sess.run(contrib.layers.l2_regularizer(0.5)(weight)))
        # l1_regularizer+l2_regularizer
        print(sess.run(contrib.layers.l1_l2_regularizer(0.5, 0.5)(weight)))
        # 损失通常被添加到
        tf.GraphKeys.REGULARIZATION_LOSSES


def logic_func():
    a1 = []
    a2 = []
    a3 = []
    cur_layer = tf.where(a1, a2, a3)
    cur_layer = tf.greater(a1, a2)
    cur_layer = tf.tanh()
    # 将一个张量中的数值限制在一个范围之内
    cur_layer = tf.clip_by_value(y, 1e-10, 1.0)


# 命名领域
# tf.name_scope(‘scope_name’)
# tf.variable_scope(‘scope_name’)
def name_scope_func():
    # 注意，这里的 with 和 python 中其他的 with 是不一样的
    # 执行完 with 里边的语句之后，这个 conv1/ 和 conv2/ 空间还是在内存中的。
    # 这时候如果再次执行上面的代码 就会再生成其他命名空间.
    # tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
    with tf.name_scope('conv1') as scope:
        weights1 = tf.Variable([1.0, 2.0], name='weights')
        bias1 = tf.Variable([0.3], name='bias')
    with tf.name_scope('conv2') as scope:
        weights2 = tf.Variable([4.0, 2.0], name='weights')
        bias2 = tf.Variable([0.33], name='bias')
    print(weights1.name)
    print(weights2.name)
    """
    conv1/weights:0
    conv2/weights:0
    第二次执行上述语句的结果为：
    conv1_1/weights:0
    conv2_1/weights:0
    """

    # 定义变量
    # tf.name_scope() 并不会对 tf.get_variable() 创建的变量有任何影响。只管理tf.Variable()
    # 只有 tf.get_variable() 创建的变量之间会发生命名冲突。和 tf.variable_scope() 配合使用，从而实现变量共享的功能。
    with tf.variable_scope('v_scope') as scope1:
        # get_variable创建变量的时候必须要提供 name 。
        # Variable 可以没有。默认Variable:0
        Weights1 = tf.get_variable('Weights', shape=[2, 3])
        bias1 = tf.get_variable('bias', shape=[3])
    # 下面来共享上面已经定义好的变量
    # note: 在下面的 scope 中的 get_variable()变量必须已经定义过了，才能设置 reuse=True.
    with tf.variable_scope('v_scope', reuse=True) as scope2:
        Weights2 = tf.get_variable('Weights')
        bias2 = tf.Variable([0.52], name='bias')
    print(Weights1.name)
    print(Weights2.name)
    print(bias2.name)
    """
    v_scope/Weights:0
    v_scope/Weights:0
    v_scope_1/bias:0
    """
    with tf.variable_scope('GRU_BACKWARD') as scope:
        num_steps = 70
        for step in range(num_steps):
            if step > 0:
                scope.reuse_variables()

    # 获取当前scope
    tf.get_variable_scope()

    # 算子的域名1
    with tf.variable_scope("foo"):
        x = 1.0 + tf.get_variable("v", [1])
    assert x.op.name == "foo/add"
    # 算子的域名2
    with tf.variable_scope("foo"):
        with tf.name_scope("bar"):
            v = tf.get_variable("v", [1])
            x = 1.0 + v
    assert v.name == "foo/v:0"
    assert x.op.name == "foo/bar/add"

    # 共享变量
    # 方式一
    # tf.get_variable_scope().reuse_variables()
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    # 方式二
    with tf.variable_scope("root"):
        # At start, the scope is not reusing.
        assert tf.get_variable_scope().reuse == False
        with tf.variable_scope("foo"):
            # Opened a sub-scope, still not reusing.
            assert tf.get_variable_scope().reuse == False
        with tf.variable_scope("foo", reuse=True):
            # Explicitly opened a reusing scope.
            assert tf.get_variable_scope().reuse == True
            with tf.variable_scope("bar"):
                # Now sub-scope inherits the reuse flag.
                assert tf.get_variable_scope().reuse == True
        # Exited the reusing scope, back to a non-reusing one.
        assert tf.get_variable_scope().reuse == False

    # scope 嵌套
    with tf.variable_scope("foo") as foo_scope:
        assert foo_scope.name == "foo"
    with tf.variable_scope("bar"):
        with tf.variable_scope("baz") as other_scope:
            assert other_scope.name == "bar/baz"
            with tf.variable_scope(foo_scope) as foo_scope2:
                assert foo_scope2.name == "foo"  # Not changed.

    # scope 域参数传递
    with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):
        v = tf.get_variable("v", [1])
        assert v.eval() == 0.4  # Default initializer as set above.
        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3))
        assert w.eval() == 0.3  # Specific initializer overrides the default.
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
            assert v.eval() == 0.4  # Inherited default initializer.
        with tf.variable_scope("baz", initializer=tf.constant_initializer(0.2)):
            v = tf.get_variable("v", [1])
            assert v.eval() == 0.2  # Changed default initializer.


# https://blog.csdn.net/lujiandong1/article/details/53385092

def save_model():
    # 3. 保存模型。
    saver = tf.train.Saver()  # 声明ta.train.Saver()类用于保存.
    save_path = saver.save(sess, 'save/filename.ckpt', global_step=5)  # 保存路径为相对路径的save文件夹,保存名为filename.ckpt
    # tf.Saver([tensors_to_be_saved]) 中可以传入一个 list，把要保存的 tensors 传入，如果没有给定这个list的话，他会默认保存当前所有的 tensors。


def save_load():
    # 初始化参数比较特殊，是一个 **kwargs
    model_to_be_restored = MyModel()  # 待恢复参数的同一模型
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored, myAwesomeOptimizer=optimizer)
    # 键名保持为“myAwesomeModel”
    save_path_with_prefix = './save/model.ckpt'
    checkpoint.save(save_path_with_prefix)
    # 恢复
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored, myAwesomeOptimizer=optimizer)
    save_path_with_prefix_and_index = './save/model.ckpt-1'
    checkpoint.restore(save_path_with_prefix_and_index)
    # or
    checkpoint.restore(tf.train.latest_checkpoint('./save'))


def load_model1():
    # 2. 加载模型。
    # 导入模型之前，必须重新再定义一遍变量。
    # 所定义的变量一定要在 checkpoint 中存在；但不是所有在checkpoint中的变量，你都要重新定义。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'save/filename.ckpt')  # 从保存路径读取


def load_model2():
    # 方式一
    saver = tf.train.import_meta_graph('save/filename.meta')
    saver.restore(tf.get_default_session(), 'save/filename.ckpt-16000')


def load_model3():
    # 方式三
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        """Load model from a checkpoint."""
        start_time = time.time()
        try:
            model.saver.restore(session, latest_ckpt)
        except tf.errors.NotFoundError as e:
            utils.print_out("Can't load checkpoint")
            print_variables_in_ckpt(latest_ckpt)
            utils.print_out("%s" % str(e))

        session.run(tf.tables_initializer())
        utils.print_out(
            "  loaded %s model parameters from %s, time %.2fs" %
            (name, ckpt_path, time.time() - start_time))
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                        (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


def load_and_add():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create some variables.
    v1 = tf.Variable([11.0, 16.3], name="v1")

    # ** 导入训练好的模型
    saver = tf.train.Saver()
    ckpt_path = '../ckpt/test-model.ckpt'
    saver.restore(sess, ckpt_path + '-' + str(1))
    print(sess.run(v1))

    # ** 定义新的变量并单独初始化新定义的变量
    v3 = tf.Variable(666, name='v3', dtype=tf.int32)
    init_new = tf.variables_initializer([v3])
    sess.run(init_new)
    # 。。。这里就可以进行 fine-tune 了
    print(sess.run(v1))
    print(sess.run(v3))

    # ** 保存新的模型。
    #  注意！注意！注意！ 一定一定一定要重新定义 saver, 这样才能把 v3 添加到 checkpoint 中
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path, global_step=2)


def tf_ckpt_info():
    # 1. 查看结构矩阵的权重信息。
    # code for finall ckpt
    # checkpoint_path = os.path.join('~/tensorflowTraining/ResNet/model', "model.ckpt")
    # code for designated ckpt, change 3890 to your num
    checkpoint_path = os.path.join('~/tensorflowTraining/ResNet/model', "model.ckpt-3890")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))


# 查看加载模型名字
def model_name():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model_saver = tf.train.Saver(tf.global_variables())
        model_ckpt = tf.train.get_checkpoint_state("model1/save/path")
        model_saver.restore(sess, model_ckpt.model_checkpoint_path)
        # 查看加载模型名字1
        op = sess.graph.get_operations()
        for m in op:
            print(m.values())
        # 查看加载模型名字2
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name)
        model_saver.restore(sess, "save_path")

        x_test_batch = []
        results = sess.run("predicty:0", {"input:0": x_test_batch})
        print(results)


# 多模型加载
def load_one_layer():
    g1 = tf.Graph()  # 加载到Session 1的graph
    feature = g1.get_tensor_by_name("dense_1").outputs[0]  # 获取某一层
    batch_predictions, batch_feature = sess.run([predictions, feature], {input_x: x_test_batch, dropout_keep_prob: 1.0})


# 多模型加载
def load_multi_graph():
    g1 = tf.Graph()  # 加载到Session 1的graph
    g2 = tf.Graph()  # 加载到Session 2的graph
    sess1 = tf.Session(graph=g1)  # Session1
    sess2 = tf.Session(graph=g2)  # Session2
    # 加载第一个模型
    # as_default使session在离开的时候并不关闭。手动关闭
    with sess1.as_default():
        with g1.as_default():
            tf.global_variables_initializer().run()
            model_saver = tf.train.Saver(tf.global_variables())
            model_ckpt = tf.train.get_checkpoint_state("model1/save/path")
            model_saver.restore(sess, model_ckpt.model_checkpoint_path)
    # 加载第二个模型
    with sess2.as_default():  # 1
        with g2.as_default():
            tf.global_variables_initializer().run()
            model_saver = tf.train.Saver(tf.global_variables())
            model_ckpt = tf.train.get_checkpoint_state("model2/save/path")
            model_saver.restore(sess, model_ckpt.model_checkpoint_path)
    # 使用的时候
    with sess1.as_default():
        with sess1.graph.as_default():  # 2
            pass
    with sess2.as_default():
        with sess2.graph.as_default():
            pass
    # 关闭sess
    sess1.close()
    sess2.close()


# 加载 + 变量固化 + 本地保存
def load_varia_save_constant():
    from tensorflow.python.framework import graph_util
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    # 另一种方式
    from tensorflow.tools.graph_transforms import TransformGraph
    from tensorflow.python.framework import graph_io

    transforms = ["quantize_weights", "quantize_nodes"]
    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
    constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, args.output_model_file, as_text=False)


# 加载固化模型
def load_frozen_graph(frozen_graph_filename):
    """
    使用方法：
    graph = load_graph(args.frozen_model_filename) 
    for op in graph.get_operations():
        print(op.name,op.values())

        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    #操作有:prefix/Placeholder/inputs_placeholder
    #操作有:prefix/Accuracy/predictions
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
        })
    :param frozen_graph_filename: 
    :return: 
    """
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def load_freeze_model2(pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print(tensors)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            op = sess.graph.get_operations()
            for m in op:
                print(m.values())
            input_x = sess.graph.get_tensor_by_name("convolution2d_1_input:0")  # 具体名称看上一段代码的input.name
            print(input_x)
            out_softmax = sess.graph.get_tensor_by_name("activation_4/Softmax:0")  # 具体名称看上一段代码的output.name
            print(out_softmax)


# tensorboard
def tensorboard_func():
    """TensorBoard 简单例子。
    tf.summary.scalar('var_name', var)        # 记录标量的变化
    tf.summary.histogram('vec_name', vec)     # 记录向量或者矩阵，tensor的数值分布变化。

    merged = tf.summary.merge_all()           # 把所有的记录并把他们写到 log_dir 中
    train_writer = tf.summary.FileWriter(log_dir + '/add_example', sess.graph)  # 保存位置

    运行完后，在命令行中输入 tensorboard --logdir=log_dir_path(你保存到log路径)
    """
    import shutil
    import numpy  as np

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # 删掉以前的summary，以免重合
    log_dir = 'summary/graph/'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    print('created log_dir path')

    a = tf.placeholder(dtype=tf.float32, shape=[100, 1], name='a')

    with tf.name_scope('add_example'):
        b = tf.Variable(tf.truncated_normal([100, 1], mean=-0.5, stddev=1.0), name='var_b')
        tf.summary.histogram('b_hist', b)
        increase_b = tf.assign(b, b + 0.2)
        c = tf.add(a, b)
        tf.summary.histogram('c_hist', c)
        c_mean = tf.reduce_mean(c)
        tf.summary.scalar('c_mean', c_mean)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)  # 保存位置
    test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

    # 每步改变一次 b 的值
    sess.run(tf.global_variables_initializer())
    for step in xrange(500):
        if (step + 1) % 10 == 0:
            _a = np.random.randn(100, 1)
            summary, _ = sess.run([merged, increase_b], feed_dict={a: _a})  # 每步改变一次 b 的值
            test_writer.add_summary(summary, step)
        else:
            _a = np.random.randn(100, 1) + step * 0.2
            summary, _ = sess.run([merged, increase_b], feed_dict={a: _a})  # 每步改变一次 b 的值
            train_writer.add_summary(summary, step)
    train_writer.close()
    test_writer.close()
    print('END!')


# example of tensorflow
def demo_tensorflow():
    import tensorflow as tf
    import numpy as np

    def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
        # add one more layer and return the output of this layer
        layer_name = 'layer%s' % n_layer
        with tf.name_scope('layer'):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
                tf.summary.histogram(layer_name + '/weights', Weights)

            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
                tf.summary.histogram(layer_name + '/biases', biases)

            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, Weights) + biases

            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)

            tf.summary.histogram(layer_name + '/outputs', outputs)

        return outputs

    # make some real data
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # define placeholder for inputs to network
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    # add hidden layer
    l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
    # add output layer
    prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

    # the error between prediction and real data
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                            reduction_indices=[1]))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test")

    # tf.global_variables_initializer().run(init)
    sess.run(init)

    # training
    saver = tf.train.Saver()
    for i in range(1000):
        summary = sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            train_writer.add_summary(result, i)
            saver.save(sess, "logs/model.ckpt", i)
            print("Adding run metadata for", i)
    train_writer.close()
    test_writer.close()


# 性能提升
def timeline_test():
    from tensorflow.python.client import timeline

    x = tf.random_normal([1000, 1000])  # 随机矩阵1000*1000
    y = tf.random_normal([1000, 1000])
    res = tf.matmul(x, y)

    # Run the graph with full trace option
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(res, options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
            # chrome://tracing


def cudnn_test():
    # 只能用于相同的序列长度
    # 没有GPU自动切换到LSTMBlockFused
    # gpu下NativeLSTM快
    # cpu下StandardLSTM BasicLSTM 快,
    from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
    cudnn_cell_fw = cudnn_rnn.CudnnLSTM(num_layers=1,
                                        num_units=hidden_state_size,
                                        direction=cudnn_rnn.CUDNN_RNN_UNIDIRECTION,
                                        input_model=cudnn_rnn.CUDNN_INPUT_LINEAR_MODE,
                                        type=tf.float32)
    outputs_fw, (h_fw, c_fw) = cudnn_cell_fw(inputs=input)


if __name__ == '__main__':
    # timeline_test()
    tf_ckpt_info()
