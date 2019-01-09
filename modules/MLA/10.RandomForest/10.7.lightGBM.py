import lightgbm as lgb
import pandas as pd
import matplotlib as plt
from sklearn import model_selection


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "min_child_weight": 50,
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',  # 设置提升类型
    #     'objective': 'regression',  # 目标函数
    #     'metric': {'l2', 'auc'},  # 评估函数
    #     'num_leaves': 31,  # 叶子节点数
    #     'learning_rate': 0.05,  # 学习速率
    #     'feature_fraction': 0.9,  # 建树的特征选择比例
    #     'bagging_fraction': 0.8,  # 建树的样本采样比例
    #     'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    #     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    # }
    # param = {'num_leaves': 111,
    #          'min_data_in_leaf': 149,
    #          'objective': 'regression',
    #          'max_depth': 9,
    #          'learning_rate': 0.005,
    #          "boosting": "gbdt",
    #          "feature_fraction": 0.7522,
    #          "bagging_freq": 1,
    #          "bagging_fraction": 0.7083,
    #          "bagging_seed": 11,
    #          "metric": 'rmse',
    #          "lambda_l1": 0.2634,
    #          "random_state": 133,
    #          "verbosity": -1}
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


target_col = "score"
cols_to_use = ["feature_a", "feature_b"]
train_df = pd.DataFrame()
test_df = pd.DataFrame()
train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = train_df[target_col].values

pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_X.loc[dev_index, :], train_X.loc[val_index, :]
    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
pred_test /= 5.

fig, ax = plt.subplots(figsize=(12, 10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
