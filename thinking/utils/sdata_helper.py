import numpy as np


# 用于产生batch
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size  # 每个epoch中包含的batch数量
    for epoch in range(num_epochs):
        # 每个epoch是否进行shuflle
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch + 1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
