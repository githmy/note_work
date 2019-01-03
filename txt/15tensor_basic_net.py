import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


def main(rnn_size, layer_size, encoder_vocab_size, decoder_vocab_size, embedding_dim, grad_clip, is_inference=False):
    # 1. 埋入层
    input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
    with tf.variable_scope('embedding'):
        encoder_embedding = tf.Variable(tf.truncated_normal(shape=[encoder_vocab_size, embedding_dim], stddev=0.1),
                                        name='encoder_embedding')
    with tf.device('/cpu:0'):
        input_x_embedded = tf.nn.embedding_lookup(encoder_embedding, input_x)

    # 2. 基本单元定义
    lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
    # 3. 多层单元定义
    encoder = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    # 4. 动态输入序列 编码
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, dtype=tf.float32)
    encoder_outputs.rnn_output

    # 5. 动态输入序列 普通解码
    decoder_scope = 6
    target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
    decoder_embedding = tf.Variable(tf.truncated_normal(shape=[encoder_vocab_size, embedding_dim], stddev=0.1),
                                    name='encoder_embedding')
    fc_layer = Dense(decoder_vocab_size)
    decoder_cell = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder_cell,
        maximum_iterations=maximum_iterations,
        output_time_major=True,
        swap_memory=True,
        scope=decoder_scope)
    outputs.rnn_output
    outputs.sample_id

    # 6. 动态输入序列 cell + help 解码
    decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
    with tf.device('/cpu:0'):
        target_embeddeds = tf.nn.embedding_lookup(decoder_embedding, target_ids)
    helper = TrainingHelper(target_embeddeds, decoder_seq_length)
    """
    BasicDecoder 结合 CustomHelper 定义自己的网络，再使用tf.contrib.seq2seq.dynamic_decode执行decode，
    最终返回：(final_outputs, final_state, final_sequence_lengths)
    CustomHelper 参数：
    initialize_fn：返回finished，next_inputs。其中finished不是scala，是一个一维向量。这个函数即获取第一个时间节点的输入。
    sample_fn：接收参数(time, outputs, state)
    返回sample_ids。即，根据每个cell的输出，如何sample。
    next_inputs_fn：接收参数(time, outputs, state, sample_ids)
    返回(finished, next_inputs, next_state)，根据上一个时刻的输出，决定下一个时刻的输入。
    """

    def initial_fn():
        initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
        initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
        return initial_elements_finished, initial_input

    def sample_fn(time, outputs, state):
        # 选择logit最大的下标作为sample
        prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
        return prediction_id

    def next_inputs_fn(time, outputs, state, sample_ids):
        # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
        pred_embedding = tf.nn.embedding_lookup(embeddings, sample_ids)
        # 输入是h_i+o_{i-1}+c_i
        next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
        elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
        all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
        next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
        next_state = state
        return elements_finished, next_inputs, next_state

    decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

    # 7. attention 编码输出到解码处理
    # Only generate alignment in greedy INFER mode.
    num_units = 20
    # memory : [batch_size, max_time, context_dim] 变换成  [batch_size, max_time, num_units]
    memory = encoder_outputs
    source_sequence_length = 30
    # 记忆长度，默认为序列长度
    memory_sequence_length = source_sequence_length
    # 选择rnn 机制
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=memory_sequence_length
    )
    # 直接使用一层或多层rnn_cell，最后输出除了lstm tuple,还额外存储了attention，time等信息。
    att_wrapper = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell,
        attention_mechanism,
        # 如果是None，则直接返回对应attention mechanism计算得到的加权和向量；
        # 如果不是None，则在调用_compute_attention方法时，得到的加权和向量还会与output进行concat，
        # 然后再经过一个线性映射，变成维度为attention_layer_size的向量
        attention_layer_size=num_units,
        # 是否将之前每一步的alignment存储在state中，主要用于后期的可视化，关注attention的关注点。
        alignment_history=True,
        # output_attention 是否返回attention，如果为False则直接返回rnn cell的输出
        output_attention=True,
        name="attention")

    # 使用attention
    batch_size = 32
    decode_time_step = 67
    # 生成全0的初始状态
    states = att_wrapper.zeros_state(batch_size, tf.float32)
    with tf.variable_scope("SCOPE", reuse=tf.AUTO_REUSE):
        for i in range(decode_time_step):
            h_bar_without_tanh, states = att_wrapper(_X, states)
            h_bar = tf.tanh(h_bar_without_tanh)
            _X = tf.nn.softmax(tf.matmul(h_bar, W), 1)

    # 8. attention 编码输出到解码处理

    # 9. 求损失
    tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
    # logits：尺寸[batch_size, sequence_length, num_decoder_symbols]
    # targets：尺寸[batch_size, sequence_length]，不用做one_hot。
    # weights：[batch_size, sequence_length]，即mask，滤去padding的loss计算，使loss计算更准确。

    # 10. batch normalize
    def batch_norm_training():
        # Calculate the mean and variance for the data coming out of this layer's linear-combination step.
        # The [0] defines an array of axes to calculate over.
        weights = tf.Variable(initial_weights)
        linear_output = tf.matmul(layer_in, weights)
        num_out_nodes = initial_weights.shape[-1]
        # Batch normalization adds additional trainable variables:
        # gamma (for scaling) and beta (for shifting).
        gamma = tf.Variable(tf.ones([num_out_nodes]))
        beta = tf.Variable(tf.zeros([num_out_nodes]))
        # These variables will store the mean and variance for this layer over the entire training set,
        # which we assume represents the general population distribution.
        # By setting `trainable=False`, we tell TensorFlow not to modify these variables during
        # back propagation. Instead, we will assign values to these variables ourselves.
        pop_mean = tf.Variable(tf.zeros([num_out_nodes]), trainable=False)
        pop_variance = tf.Variable(tf.ones([num_out_nodes]), trainable=False)
        batch_mean, batch_variance = tf.nn.moments(linear_output, [0])
        # Calculate a moving average of the training data's mean and variance while training.
        # These will be used during inference.
        # Decay should be some number less than 1. tf.layers.batch_normalization uses the parameter
        # "momentum" to accomplish this and defaults it to 0.99
        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

        # The 'tf.control_dependencies' context tells TensorFlow it must calculate 'train_mean'
        # and 'train_variance' before it calculates the 'tf.nn.batch_normalization' layer.
        # This is necessary because the those two operations are not actually in the graph
        # connecting the linear_output and batch_normalization layers,
        # so TensorFlow would otherwise just skip them.
        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(linear_output, batch_mean, batch_variance, beta, gamma, epsilon)

    def batch_norm_inference():
        # During inference, use the our estimated population mean and variance to normalize the layer
        return tf.nn.batch_normalization(linear_output, pop_mean, pop_variance, beta, gamma, epsilon)

    def crf_funcs():
        with tf.name_scope('CRF'):
            # inputs: [batch_size, max_seq_len, num_tags]
            # tag_indices: [batch_size, max_seq_len]
            # seq_length: [batch_size]
            # log_likelihood: scalar
            # transition_params: [num_tags, num_tags]
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs, tag_indices, seq_length)

            # 作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
            # inputs: 一个形状为[seq_len, num_tags] matrix of unary potentials.
            # transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            # viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表.
            # viterbi_score: A float containing the score for the Viterbi sequence.
            viterbi,viterbi_score = tf.contrib.crf.viterbi_decode(inputs, transition_params)

            # 进行解码（维特比算法），获得解码之后的序列viterbi_sequence和分数viterbi_score
            # inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            # transition_params: 一个形状为[num_tags, num_tags] 的转移矩阵
            # sequence_length: 一个形状为[batch_size] 的 ,表示batch中每个序列的长度
            # decode_tags:一个形状为[batch_size, max_seq_len] 的tensor,类型是tf.int32.表示最好的序列标记.
            # best_score: 有个形状为[batch_size] 的tensor, 包含每个序列解码标签的分数.
            predictions, viterbi_score = tf.contrib.crf.crf_decode(
                inputs, transition_params, seq_length)


    # tain是选项
    if self.use_batch_norm:
        # If we don't include the update ops as dependencies on the train step, the
        # tf.layers.batch_normalization layers won't update their population statistics,
        # which will cause the model to fail at inference time
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    else:
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)