# coding:utf-8
import pandas as pd
import numpy as np
import re
import json
import os


def Loss(theta, X_b, y):
    '''
    损失函数
    '''
    return np.sum((y - np.dot(X_b, theta)) ** 2) / len(y)


def dLoss(theta, X_b, y):
    '''
    损失函数对theta的偏导数
    '''
    gradient = X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)
    return gradient


def gradient_decent(theta, X_b, y):
    '''
    梯度下降过程
    '''
    eta = 0.01  # eta代表是学习速率
    episilon = 1e-8  # episilon用来判断损失函数是否收敛
    while True:
        last_theta = theta
        theta = theta - eta * dLoss(theta, X_b, y)
        if abs(Loss(theta, X_b, y) - Loss(last_theta, X_b, y)) <= episilon:  # 判断损失函数是否收敛，也可以限定最大迭代次数
            break
    return theta


def dLoss_sgd(theta, X_b_i, y_i):
    '''
    单样本随机梯度下降，损失函数对theta的偏导数
    '''
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2


def sgd(X_b, y, theta, n_iters):
    '''
    随机梯度下降过程
    '''
    t0 = 5
    t1 = 50

    def learn_rate(t):
        return t0 / (t + t1)

    theta = theta
    for cur_iter in range(n_iters):
        rand_i = np.random.randint(len(X_b))
        gradient = dLoss_sgd(theta, X_b[rand_i], y[rand_i])
        theta = theta - learn_rate(cur_iter) * gradient

    return theta


def dLoss_mbgd(theta, X_b_n, y_n, num):
    '''
    小批量随机梯度下降，损失函数对theta的偏导数
    '''
    return X_b_n.T.dot(X_b_n.dot(theta) - y_n) * 2 / num


def mbgd(theta, X_b, y, num, n_iters):
    '''
    小批量随机梯度下降过程
    '''
    t0 = 5
    t1 = 50
    theta = theta
    num = num

    def learn_rate(t):
        return t0 / (t + t1)

    for cur_iter in range(n_iters):
        x_index = np.random.randint(0, len(y), num)
        gradient = dLoss_mbgd(theta, X_b[x_index,], y[x_index], num)
        theta = theta - learn_rate(cur_iter) * gradient

    return theta


def main():
    X = 2 * np.random.random(size=20000).reshape(-1, 2)
    y = X[:, 0] * 2. + X[:, 1] * 3. + 4. + np.random.normal(size=10000)
    temp = np.ones((len(y), 1))
    X_b = np.hstack((temp, X))  # 为了矩阵运算方便在X中加上全为1的一列
    theta = np.zeros(X_b.shape[1])  # theta是参数，梯度下降通过不断更新theta的值使损失函数达到最小值
    print(X_b.shape)
    print(y.shape)
    print(theta.shape)
    print(mbgd(theta, X_b, y, num=20, n_iters=len(X_b) // 3))
    print(sgd(X_b, y, theta, n_iters=len(X_b) // 3))
    rst = gradient_decent(theta, X_b, y)
    print(rst)
    pass


if __name__ == "__main__":
    main()
