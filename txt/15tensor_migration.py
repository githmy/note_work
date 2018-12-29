import numpy as np
import tensorflow as tf
import os

BASE_DIR = "./"


# 数据集下载
# wget http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz

# 先将no_top部分拆开，作为特征提取部分的卷积层，在用数量很少的训练集训练时参数就不允许被修改了
def cnn_model_no_top(features, trainable):
    """
    :param features: 原始输入
    :param mode: estimator模式
    :param trainable: 该层的变量是否可训练
    :return: 不含最上层全连接层的模型
    """
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                             trainable=trainable)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                             trainable=trainable)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
    return pool2_flat


# 正式模型
def cnn_model_fn(features, labels, mode, params):
    """
    用于构造estimator的model_fn
    :param features: 输入
    :param labels: 标签
    :param mode: 模式
    :param params: 用于迁移学习和微调训练的参数
        nb_classes
        transfer
        finetune
        checkpoints
        learning_rate
    :return: EstimatorSpec
    """
    logits_name = "predictions"
    # 把labels转换成ont-hot 形式
    labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params["nb_classes"])
    # 迁移学习不允许修改底层的参数
    model_no_top = cnn_model_no_top(features["x"], trainable=not (params.get("transfer") or params.get("finetune")))
    with tf.name_scope("finetune"):
        # 此层在第二次迁移学习时允许修改参数，将第二次迁移学习称作微调了
        dense = tf.layers.dense(inputs=model_no_top, units=1024, activation=tf.nn.relu,
                                trainable=params.get("finetune"))
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # 最上层任何训练都可以修改参数
    logits = tf.layers.dense(inputs=dropout, units=params.get("nb_classes"), name=logits_name)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 使用softmax交叉熵作为损失函数
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 加载已有的存档点参数的方法
        if params.get("checkpoints") and isinstance(params.get("checkpoints"), (tuple, list)):
            for ckpt in params.get("checkpoints"):
                # [0]是存档点路径，[1]为是否加载倒数第二个全连接层参数
                if ckpt[1]:
                    print("restoring base ckpt")
                    variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=[logits_name])
                    tf.train.init_from_checkpoint(ckpt[0], {v.name.split(':')[0]: v for v in variables_to_restore})
                    print("restored base ckpt")
                else:
                    print("restoring transferred ckpt")
                    variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=[logits_name,
                                                                                             "finetune", ])
                    tf.train.init_from_checkpoint(ckpt[0], {v.name.split(':')[0]: v for v in variables_to_restore})
                    print("restored transferred ckpt")

        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=params.get("learning_rate", 0.0001))
        train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                        predictions=predictions['classes'],
                                        name='accuracy')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


# 数据处理函数
import os
from functools import partial
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小，再normalize
def load_image_tf(filename, label, height, width, channels=3):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels)
    image_decoded.set_shape([None, None, None])
    image_decoded = tf.image.central_crop(image_decoded, 1)
    image_decoded = tf.image.resize_images(image_decoded, tf.constant([height, width], tf.int32),
                                           method=ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, height, width)
    image_resized = tf.reshape(image_resized, [height, width, channels])
    image_resized = tf.divide(image_resized, 255)
    image_resized = tf.subtract(image_resized, 0.5)
    image_resized = tf.multiply(image_resized, 2.)
    return image_resized, label


def read_folder(folders, labels):
    if not isinstance(folders, (list, tuple, set)):
        raise ValueError("folders 应为list 或 tuple")
    all_files = []
    all_labels = []
    for i, f in enumerate(folders):
        files = os.listdir(f)
        for file in files:
            all_files.append(os.path.join(f, file))
            all_labels.append(labels[i])
    dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
    return dataset, len(all_files)


def dataset_input_fn(folders, labels, epoch, batch_size,
                     height, width, channels,
                     scope_name="dataset_input",
                     feature_name=None):
    def fn():
        with tf.name_scope(scope_name):
            dataset, l = read_folder(folders, labels)
            dataset = dataset.map(partial(load_image_tf, height=height, width=width, channels=channels))
            dataset = dataset.shuffle(buffer_size=l).repeat(epoch).batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            one_element = iterator.get_next()
            if feature_name:
                return {str(feature_name): one_element[0]}, one_element[1]
            return one_element[0], one_element[1]

    return fn


# 开始训练
# 首先准备训练数据和验证数据
if __name__ == '__main__':
    # 训练MNIST的estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_model", params={
        "nb_classes": 10,
    })
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # 训练
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=np.asarray(train_labels),
        batch_size=50, num_epochs=50, shuffle=True
    )

    mnist_classifier.train(train_input_fn)
    # 验证
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=np.asarray(eval_labels),
        batch_size=50, num_epochs=2, shuffle=True
    )

    eval_result = mnist_classifier.evaluate(eval_input_fn)
    print(eval_result)  # MNIST数据集上的准确率

    # 第一次迁移学习，即只重新训练最上层的全连接层
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_transfer_model", params={
        "transfer": True,
        "nb_classes": 2,
        "checkpoints": [
            (os.path.join(BASE_DIR, "models", "mnist", "mnist_model"), True),  # (dir, load_dense_layer)
        ]
    })

    train_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="train",
        epoch=100, batch_size=50,
        feature_name="x"
    )

    mnist_classifier.train(train_input_fn)

    eval_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="eval",
        epoch=1, batch_size=50,
        feature_name="x"
    )
    result = mnist_classifier.evaluate(eval_input_fn)
    print(result)

    # 第二次迁移学习，训练所有全连接层
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_finetune_model", params={
        "finetune": True,
        "nb_classes": 2,
        "checkpoints": [(os.path.join(BASE_DIR, "models", "mnist", "mnist_model"), True),  # (dir, load_dense_layer)
                        (os.path.join(BASE_DIR, "models", "mnist", "mnist_transfer_model"), False)
                        ]
    })

    train_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="train",
        epoch=100, batch_size=50,
        feature_name="x"
    )

    mnist_classifier.train(train_input_fn)

    eval_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="eval",
        epoch=1, batch_size=50,
        feature_name="x"
    )
    result = mnist_classifier.evaluate(eval_input_fn)
    print(result)
