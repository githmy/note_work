#
# scipy(>=0.13.3)
# numpy(>=1.8.2)
# scikit-learn(>=0.19.0)
#
# pip3 install -U seglearn
#
# pip install -U git+https://github.com/dmbee/seglearn.git
#

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

from statsmodels.tsa.arima_model import ARIMA

# (p,d,q)
# 1、p - 部分自相关函数表第一次截断的上层置信区间是滞后值。如果你仔细看，该值是p = 2。
# 2、q - 自相关函数表第一次截断的上层置信区间是滞后值。如果你仔细看，该值是q = 2。
model = ARIMA(ts_log, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
