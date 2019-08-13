import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.anova as anova
from statsmodels.formula.api import ols
import statsmodels.formula.api as sm


def ols_model():
    # 1. 单因素方差分析
    # 常规最小方差
    pddata = pd.read_csv()
    model = ols('因变量-C(自变量)', data=pddata.dropna()).fit()
    table1 = anova.anova_lm(model)
    # p值小于0.05 说明意外少，既同步相关
    print(table1)
    # 2. 多因素方差分析
    model = ols('因变量-C(削弱的自变量)+C(增强的自变量)', data=pddata.dropna()).fit()
    table2 = anova.anova_lm(model)
    # p值小于0.05 说明意外少，既同步相关
    # 3. 析因方差分析
    model = ols('因变量-C(自变量1)*C(自变量2)', data=pddata.dropna()).fit()
    table3 = anova.anova_lm(model)
    # p值大于0.05 说明不是偶然，既不同步，不相关


def regression_model():
    pddata = pd.read_csv()
    shindex = pddata[pddata.indexcd == 1]
    szindex = pddata[pddata.indexcd == 399106]
    model = sm.ols('因变量-C(自变量)', data=pddata.dropna()).fit()



def main(args=None):
    pass
    # 1. 命令行
    # parajson = get_paras(args)


if __name__ == '__main__':
    # 1. 参数解析
    main(sys.argv[1:])
    # deep_network()
    # navigation()
