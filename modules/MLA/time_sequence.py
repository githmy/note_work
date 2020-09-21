# coding:utf-8
# https://zhuanlan.zhihu.com/p/158871695
import matplotlib.pyplot as plt


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


def 配对交易策略():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import coint
    from scipy.stats import pearsonr, spearmanr

    data = pd.read_excel(r'futures.xlsx')

    # 测试相关性和协整关系
    _, pv_coint, _ = coint(data['CU.SHF'], data['SF.CZC'])
    corr, pv_corr = pearsonr(data['CU.SHF'], data['SF.CZC'])
    print("Cointegration pvalue : %0.4f" % pv_coint)
    print("Correlation coefficient is %0.4f and pvalue is %0.4f" % (corr, pv_corr))
    # 画出结算价走势
    plotSettlePrice(data, 'CU.SHF', 'SF.CZC', 'CU.SHF铜', 'SF.CZC硅铁',
                    'CU.SHF铜 和 SF.CZC硅铁 价格走势图')
    # 计算CU.SHF结算价/SF.CZC结算价的比例Ratios
    S1, S2, ratios = getRatios(data, 'CU.SHF', 'SF.CZC', 1)
    # ZScore
    zScore = getZScore(ratios, 1)
    # 特征工程
    train, test, zscore_mv = getMovingIndex(ratios, 0.7, 5, 60, 1)
    # 找到交易信号
    buy, sell = getTradeSignal(train, zscore_mv, 60, 1, ratios)
    # 两个合约具体交易
    Trade2Contract(data, 'CU.SHF', 'SF.CZC', buy, sell, 60)
    # 总体
    print(PairsTrade(S1, S2, 5, 60))


def plotSettlePrice(df, var1, var2, var1_name, var2_name, title):
    " 画出结算价走势 "
    temp = df[[var1, var2]].dropna()
    fig = plt.figure(figsize=(10, 5))
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    ax1 = fig.add_subplot(111)
    ax1.plot(temp[var1], c='blue')
    ax1.set_ylabel(var1_name)
    ax1.set_title(title)
    plt.legend(loc='upper left', labels=[var1_name])
    ax1.set_xlabel('年份')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(temp[var2], c='orange')
    ax2.set_ylabel(var2_name)
    ax2.set_xlabel('年份')
    plt.legend(loc='upper right', labels=[var2_name])
    plt.show()


def getRatios(df, var1, var2, plotOrNot):
    " 计算CU.SHF结算价/SF.CZC结算价的比例Ratios "
    df1 = df[[var1, var2]].dropna()
    S1 = df1[var1]
    S2 = df1[var2]
    ratios = S1 / S2
    if plotOrNot:
        plt.figure(figsize=(10, 5))
        ratios.hist(bins=200)
        plt.title("Ratios histogram")
        plt.ylabel('Frequency')
        plt.xlabel('Intervals')
        plt.show()
    return S1, S2, ratios


def getZScore(ratios, plotOrNot):
    """
    ZScore将交易信号Ratios标准化处理。
    为什么要标准化处理？
    因为我们需要通过价格比例Ratios找到交易信号。但是Ratios的分布取决于具体的合约；而我们希望任何两个合约输入到程序中，都能够按照固定的模式找到交易信号。
    怎样才是触发交易的信号？Ratios具体在哪个上限以上、才说明了CU更贵、SF更便宜，下降到哪个下限以下、才说明了相反情况？
    因此，我们把Ratios标准化为Z-Score，并且认为在一个标准差(-1,1)内的波动是正常的；超过一个标准差(-∞,-1)&(1, ∞)即为触发了交易信号。这样，即便选取其他的合约，也不会受到β的不同而大改程序。
    画出ZScore的时间走势，同时画出±1的水平直线，如果ZScore和±1的水平直线相交，我们认为触发了交易信号。
    """
    zScore = (ratios - ratios.mean()) / ratios.std()

    if plotOrNot:
        zScore.plot(figsize=(10, 5))
        plt.axhline(zScore.mean(), color='black')
        plt.axhline(1.0, color='red', linestyle='--')
        plt.axhline(-1.0, color='green', linestyle='--')
        plt.legend(['Ratio z-score', 'Mean', '+1', '-1'])
        plt.title("zScore time series")
        plt.xlabel('Date')
        plt.ylabel('Intervals')
        plt.show()

        plt.figure(figsize=(10, 5))
        zScore.hist(bins=200)
        plt.title("zScore histogram")
        plt.ylabel('Frequency')
        plt.xlabel('Intervals')
        plt.show()
    return zScore


def getMovingIndex(ratios, train_pct, w1, w2, plotOrNot):
    """
    ZScore作为信号过于薄弱。因为它集中的是“整个时期”的关系；而我们交易必然是一个动态变化的过程。这里需要不断计算“一段时间”的ZScore，而不是“整个时期”的ZScore。
    例如，在1月1号~2月1号的区间，我们得到一个ZScore，用于得到2月2号的交易信号；然后，在在1月2号~2月2号的区间，再次得到一个ZScore，用于得到2月3号的交易信号；以此类推；这是一个滚动的过程。
    因此，我们滚动计算一段区间的ZScore。这里，我选取了Ratios滚动5日的均值、Ratios滚动60日的均值以及Ratios滚动60日的标准差，计算真正的信号ZScore_moving。
    这里5、60的区间能够通过机器学习的算法进行调整，得到最好的结果的滚动区间。例如GPlearn。
    同时拆分训练集和测试集。
    """
    ### w1 < w2，拆分训练集+验证集，训练集的比例是train_pct
    length = len(ratios)
    trainLength = int(train_pct * length)
    train = ratios[:trainLength]
    test = ratios[trainLength:]

    # 计算指标moving_average, moving_std，以及moving_z_score：这里可以使用gplearn！！
    # 希望通过moving_z_score找到信号
    ratios_mavg1 = train.rolling(window=w1, center=False).mean()
    ratios_mavg2 = train.rolling(window=w2, center=False).mean()
    std = train.rolling(window=w2, center=False).std()
    zscore_mv = (ratios_mavg1 - ratios_mavg2) / std

    if plotOrNot:
        plt.figure(figsize=(10, 5))
        zscore_mv.hist(bins=200)
        plt.title("zScore with signals histogram")
        plt.ylabel('Frequency')
        plt.xlabel('Intervals')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(train.index, train.values)
        plt.plot(ratios_mavg1.index, ratios_mavg1.values)
        plt.plot(ratios_mavg2.index, ratios_mavg2.values)
        plt.legend(['Ratio', '%dd Ratio MA' % w1, '%dd Ratio MA' % w2])
        plt.ylabel('Ratio')
        plt.show()

        plt.figure(figsize=(10, 5))
        zscore_mv.plot()
        plt.axhline(0, color='black')
        plt.axhline(1.0, color='red', linestyle='--')
        plt.axhline(-1.0, color='green', linestyle='--')
        plt.legend(loc='upper right', labels=['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
        plt.show()
    return train, test, zscore_mv


def getTradeSignal(train, zscore_mv, w2, plotOrNot, ratios):
    """
    在得到了ZScore_mv和±1相交的点之后，即为触发了交易信号。这里再把交易信号投射到Ratios曲线上。
    信号-1<= zscore_mv <=1，那么buy=0和sell=0；
    注意这里的buy是指“buy ratios”，sell是指“sell ratios”；注意这里定义的buy和sell的操作是同时操作两个合约。
    buy只有在信号<-1的时候出现，long CU.SHF，short SF.CZC；
    sell只有在信号>1的时候出现，short CU.SHF，long SF.CZC。
    """
    # Plot the ratios and buy and sell signals from z score
    plt.figure(figsize=(10, 5))

    train[w2:].plot()
    buy = train.copy()
    sell = train.copy()

    # 信号ratios = CU.SHF / SF.CZC，衍生出buy和sell。
    # 其他时候ratios = 0.
    buy[zscore_mv > -1] = 0
    sell[zscore_mv < 1] = 0

    if plotOrNot:
        buy[60:].plot(color='g', linestyle='None', marker='^')
        sell[60:].plot(color='r', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, ratios.min(), ratios.max()))
        plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
        plt.show()
    return buy, sell


def Trade2Contract(df, var1, var2, buy, sell, w2):
    """
    将Ratios上找到的交易信号，再次投射到具体的合约上。
    """
    S1, S2, ratios = getRatios(df, var1, var2, 0)
    plt.figure(figsize=(10, 5))
    S1 = S1.reindex(index=buy.index)
    S2 = S2.reindex(index=buy.index)
    S1[w2:].plot(color='b')
    S2[w2:].plot(color='c')

    # buyR和sellR先填充0。
    buyR = 0 * S1.copy()
    sellR = 0 * S1.copy()

    # 即buy只有在信号ratios<-1的时候保持ratios原值，此刻long S1=CU.SHF，short S2=SF.CZC
    buyR[buy != 0] = S1[buy != 0]
    sellR[buy!=0] = S2[buy!=0]

    # 即sell只有在信号ratios>1的时候保持ratios原值，此刻short S1=CU.SHF，long S2=SF.CZC。
    buyR[sell != 0] = S2[sell != 0]
    sellR[sell != 0] = S1[sell != 0]

    buyR[w2:].plot(color='g', linestyle='None', marker='^')
    sellR[w2:].plot(color='r', linestyle='None', marker='^')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(S1.min(), S2.min()),max(S1.max(), S2.max())))

    plt.legend([var1, var2, 'Buy Signal', 'Sell Signal'])
    plt.show()


def PairsTrade(S1, S2, window1, window2):
    """
    在连续日度交易的程序中，我们需要注意
    原始资金=0，通过short一方合约得到资金、再long另一方合约，因此是“空手套白狼”；
    当ZScore_mv<-1，buy "ratios"——long CU + short SF，那么CU的仓位+，SF仓位-；
    当ZScore_mv >1，sell "ratios"——short CU + long SF，那么CU的仓位-，SF仓位+；
    当ZScore_mv回落到[-0.5,0.5]之内，我们进行反向操作、立刻清仓，所有仓位清零；同时赚的钱=开仓的钱-清仓的钱。
    注意，第i日的得到的信号，用于第i+1日的交易，不要用到未来信息。 
    """
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0
    # Compute rolling mean and rolling standard deviation
    ratios = S1 / S2
    mv_ave1 = ratios.rolling(window=window1, center=False).mean()
    mv_ave2 = ratios.rolling(window=window2, center=False).mean()
    mv_std = ratios.rolling(window=window2, center=False).std()
    zscore = (mv_ave1 - mv_ave2) / mv_std

    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    length = len(ratios)
    for i in range(length - 1):
        # 如果信号zscore > 1，那么short s1（s1仓位-1）,得到的钱long s2*ratios（s1仓位+1*ratios）。
        if zscore[i] > 1:
            money += S1[i + 1] - S2[i + 1] * ratios[i + 1]
            countS1 -= 1
            countS2 += ratios[i + 1]

        # 如果信号zscore < -1，那么short ratios*s2,得到的钱long s1。
        elif zscore[i] < -1:
            money -= S1[i + 1] - S2[i + 1] * ratios[i + 1]
            countS1 += 1
            countS2 -= ratios[i + 1]

        # 如果信号zscore处在(-0.5, 0.5)之间，清仓——反向操作，用此刻的价格*此刻的仓位作为清仓的成本。同时仓位清零。
        elif abs(zscore[i]) < 0.5:
            money += countS1 * S1[i + 1] + countS2 * S2[i + 1]
            countS1 = 0
            countS2 = 0

    return money

def main():
    pass


if __name__ == "__main__":
    main()
    配对交易策略()
