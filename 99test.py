import statsmodels.api as sm
import pandas as pd
import numpy as np


def liner_demo():
    pd_data = pd.DataFrame()
    model = sm.OLS(np.log(pd_data["depend_var"]),
                   sm.add_constant(pd_data[["constant_column1", "constant_column2"]])).fit()
    print(model.summary())
    # pvalue 小于0.05的可以作为系数  y = coef * log(depend_var) + coef * constant_columns


def conponent_profit():
    import ffn
    close = pd.DataFrame()["close"]
    returns = ffn.get('aapl,msft,c,gs,ge', start='2010-01-01').to_returns(close).dropna()
    returns.calc_mean_var_weights().as_format('.2%')
    # 年化率
    simpleret = ffn.to_returns(close)
    # 复利化
    simpleret = ffn.to_log_returns(close)
    annue = (1 + simpleret).cumprod()[-1] ** (245 / 311) - 1


if __name__ == "__main__":
    liner_demo()
