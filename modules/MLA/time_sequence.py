# coding:utf-8
# https://zhuanlan.zhihu.com/p/158871695

def 相关性():
    """
    相关性（correlation）
    X 和 Y之间的线性依存关系（linear dependence）
    corr(X,Y)=0，未必独立（独立定义P(XY)=P(X)P(Y)）
    [-1,1]之间—— ±1是完全线性相关。
    pearsonr检测的是具体值之间的相关性，
    spearsonr检测的是rank之间的相关性
    :return: 
    """
    from scipy.stats import pearsonr, spearmanr
    pearson_corr, pearson_pvalue = pearsonr(X, Y)
    spearnman_corr, spearnman_pvalue = spearmanr(X, Y)

def 平稳性():
    """
    （Stationarity）
    数据产生过程的参数（例如均值和方差）不随着时间变化，那么数据平稳。
    P值越小，说明越可能是平稳的时间
    :return: 
    """
    from statsmodels.tsa.stattools import adfuller
    pvalue = adfuller(some_series)[1]

def 单整_协整():
    """
    单整阶数（order of integration）
    I(0) ：如果一个时间序列Y的移动平均表达 [公式] ，满足条件 [公式] ，那么 [公式] 服从 [公式] ；其中 [公式] 是随机过程的白噪声（残差项）， [公式] 是残差项的权重， [公式] 是起决定作用的序列。
    注意： I(1) 属于 stationarity
    I(1) ：如果一个序列X的一阶差分服从I(0) ，那么X服从I(1)。
    实际运用：资产收益率一般被认为是服从[公式] ，资产价格被认为服从[公式] 。
    :return: 
    """
    from statsmodels.tsa.stattools import coint
    _, pvalue, _ = coint(X1, X2)

def 协整VS相关性():
    """
    协整不一定相关
    协整关系非常显著p-value = 0.0；但是基本没有相关性。
    :return: 
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr
    from statsmodels.tsa.stattools import coint
    import matplotlib.pyplot as plt

    x1 = pd.Series(np.random.normal(0, 1, 1000))
    x2 = x1.copy()
    for i in range(10):
        x2[(100 * i):(100 * i + 100)] = (-1) ** i

    plt.figure(figsize=(6, 3))
    plt.plot(x1, c='blue')
    plt.plot(x2, c='red')
    plt.show()

    _, pv, _ = coint(x1, x2)
    print("Cointegration test p-value : ", pv)
    correl, pv = pearsonr(x1, x2)
    print("Correlation : ", correl)

    """
    相关不一定协整
    相关性基本接近1，但是协整关系不显著。
    配对交易策略中，协整关系是重点，相关性不用考虑。
    """
    ret1 = np.random.normal(1, 1, 100)
    ret2 = np.random.normal(2, 1, 100)
    s1 = pd.Series(np.cumsum(ret1))
    s2 = pd.Series(np.cumsum(ret2))
    plt.figure(figsize=(6, 3))
    plt.plot(s1)
    plt.plot(s2)
    plt.show()
    _, pv, _ = coint(s1, s2)
    print("Cointegration test p-value : ", pv)
    correl, pv = pearsonr(s1, s2)
    print("Correlation : ", correl)

def main():
    pass


if __name__ == "__main__":
    main()
