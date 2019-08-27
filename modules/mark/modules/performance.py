from __future__ import print_function
import pandas as pd
import numpy as np


# performance.py:评估策略效果
# 创建基于策略的Sharpe比率，基准为0
def create_sharpe_ratio(returns, periods=252):
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)


# 计算PNL的最大回撤和回撤时间
def create_drawdowns(pnl):
    """
    计算PNL的最大回撤和回撤时间
    """
    hwm = [0]
    idx = pnl.index
    drawdown = pd.Series(index=idx)
    duration = pd.Series(index=idx)
    for t in range(1, len(idx)):
        hwm.append(max(hwm[t - 1], pnl[t]))
        drawdown[t] = (hwm[t] - pnl[t])
        duration[t] = (0 if drawdown[t] == 0 else duration[t - 1] + 1)
    return drawdown, drawdown.max(), duration.max()
