from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


def plot_embedding(X, title=None):
    """
    画图展示
    :param X: 数组降维后的数据
    :param title: 标注内容
    :return: 展示图像
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def tsne(arr):
    """
    对数据进行t-sne降维
    :param arr: 传入的数组
    :return: 降维的数据
    """
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    arr_tsne = tsne.fit_transform(arr)
    return arr_tsne


if __name__ == "__main__":
    digits = datasets.load_digits(n_class=10)
    print('执行')
    X = digits.data
    print(X)

    y = digits.target
    n_samples, n_features = X.shape
    n_neighbors = 30
    arr = tsne(X)
    plot_embedding(arr)
    plt.show()
