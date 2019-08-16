import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.anova as anova
from statsmodels.formula.api import ols
import statsmodels.formula.api as sm
import ffn
from scipy.stats import norm


def ols_model():
    # 1. 单因素方差分析
    # 常规最小方差
    pddata = pd.read_csv()
    model = ols('因变量 ~C(自变量)', data=pddata.dropna()).fit()
    table1 = anova.anova_lm(model)
    # p值小于0.05 说明意外少，既同步相关
    print(table1)
    # 2. 多因素方差分析
    model = ols('因变量 ~ C(削弱的自变量)+C(增强的自变量)', data=pddata.dropna()).fit()
    table2 = anova.anova_lm(model)
    # p值小于0.05 说明意外少，既同步相关
    # 3. 析因方差分析
    model = ols('因变量 ~ C(自变量1)*C(自变量2)', data=pddata.dropna()).fit()
    table3 = anova.anova_lm(model)
    # p值大于0.05 说明不是偶然，既不同步，不相关
    # 拟合的值，残差,pearson残差
    model.fittedvalues, model.resid, model.resid_pearson


def regression_model():
    pddata = pd.read_csv()
    shindex = pddata[pddata.indexcd == 1]
    szindex = pddata[pddata.indexcd == 399106]
    model = sm.ols('因变量-C(自变量)', data=pddata.dropna()).fit()


def liner_demo():
    pd_data = pd.DataFrame()
    model = sm.OLS(np.log(pd_data["depend_var"]),
                   sm.add_constant(pd_data[["constant_column1", "constant_column2"]])).fit()
    print(model.summary())
    # pvalue 小于0.05的可以作为系数  y = coef * log(depend_var) + coef * constant_columns


def conponent_profit():
    close = pd.DataFrame()["close"]
    returns = ffn.get('aapl,msft,c,gs,ge', start='2010-01-01').to_returns(close).dropna()
    returns.calc_mean_var_weights().as_format('.2%')
    # 1. 方法
    Tesla['Return'] = (Tesla['Close'] - Tesla['Close'].shift(1)) / Tesla['Close'].shift(1)
    Tesla = Tesla.dropna()
    # 2. 方法
    GM['Return'] = ffn.to_returns(GM['Close'])
    # 3. 方法
    Ford['Return'] = Ford['Close'].pct_change(1)
    Ford = Ford.dropna()
    # 年化率
    simpleret = ffn.to_returns(close)
    # 复利化
    simpleret = ffn.to_log_returns(close)
    annue = (1 + simpleret).cumprod()[-1] ** (245 / 311) - 1
    # 方差
    simpleret.std()


def risk_func():
    pass


# 时间T内，组合损失X的置信度小于1-a%
def value_at_risk():
    # 半方差公式 只算下降的
    def cal_half_dev(returns):
        # 均值
        mu = returns.mean()
        tmp = returns["returnss" < mu]
        half_deviation = (sum((mu - tmp) ** 2) / len(returns)) ** 0.5
        return half_deviation

    close = pd.DataFrame()["close"]
    simple_return = ffn.to_returns(close)
    # 半方差的大小
    res = cal_half_dev(simple_return)
    print(res)
    # 历史模拟
    simple_return.quantile(0.05)
    # 风险值最差0.05的位置
    norm.ppf(0.05, simple_return.mean(), simple_return.std())
    # 最差0.05的均值期望
    worst_return = simple_return["returnss" < simple_return.quantile(0.05)].mean()
    # 最大回撤 展开
    price = (1 + simple_return).cumprod()  # 返回的依然是数组
    simple_return.cummax() - price
    # ...
    # 最大回撤
    ffn.calc_max_drawdown(price)
    ffn.calc_max_drawdown((1 + simple_return).cumprod())


def makviz():
    """
    效用U(simple_return)
    组分权重w
    E(U(sigma(w*simple_return)))  s.t. sigma(w)=1 
    min σ^2(simple_return)=sigma(wi^2*σi(simple_return)^2)+sigma(wi*wj*σ(simple_return i,simple_return j)) s.t. simple_return.mean=sigma(wi*E(simple_return))  
    :return: 
    """
    close = pd.DataFrame()["close"]
    # 相关性协方差 series (列2) ， DataFrame (空) 返回矩阵
    close.corr()


# Meanvariance
from scipy import linalg


class MeanVariance:
    def __init__(self, returns):
        self.returns = returns

    def minVar(self, goalRet):
        covs = np.array(self.returns.cov())
        means = np.array(self.returns.mean())
        L1 = np.append(np.append(covs.swapaxes(0, 1), [means], 0),
                       [np.ones(len(means))], 0).swapaxes(0, 1)
        L2 = list(np.ones(len(means)))
        L2.extend([0, 0])
        L3 = list(means)
        L3.extend([0, 0])
        L4 = np.array([L2, L3])
        L = np.append(L1, L4, 0)
        results = linalg.solve(L, np.append(np.zeros(len(means)), [1, goalRet], 0))
        return (np.array([list(self.returns.columns), results[:-2]]))

    def frontierCurve(self):
        goals = [x / 500000 for x in range(-100, 4000)]
        variances = list(map(lambda x: self.calVar(self.minVar(x)[1, :].astype(np.float)), goals))
        plt.plot(variances, goals)

    def meanRet(self, fracs):
        meanRisky = ffn.to_returns(self.returns).mean()
        assert len(meanRisky) == len(fracs), 'Length of fractions must be equal to number of assets'
        return (np.sum(np.multiply(meanRisky, np.array(fracs))))

    def calVar(self, fracs):
        return (np.dot(np.dot(fracs, self.returns.cov()), fracs))


# minVar = MeanVariance(sh_return)


def main(args=None):
    pass
    # 1. 命令行
    # parajson = get_paras(args)


if __name__ == '__main__':
    # 1. 参数解析
    main(sys.argv[1:])
