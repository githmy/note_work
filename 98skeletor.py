import pandas as pd
import numpy as np
import re
import json
import os

pathf = os.path.join("..", "data", "particles")


def getdata():
    trainpd = pd.read_csv(os.path.join(pathf, "train.csv"))
    print(trainpd.head(1))
    print(trainpd.shape)
    eventpd = pd.read_csv(os.path.join(pathf, "event.csv"))
    print(eventpd.head(1))
    print(eventpd.shape)
    samplepd = pd.read_csv(os.path.join(pathf, "sample.csv"))
    print(samplepd.head(1))
    print(samplepd.shape)
    testpd = pd.read_csv(os.path.join(pathf, "test.csv"))
    print(testpd.head(1))
    print(testpd.shape)
    exit()
    return testpd


def stacking():
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.datasets.samples_generator import make_blobs

    '''创建训练的数据集'''
    data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)

    '''模型融合中使用到的各个单模型'''
    clfs = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]

    '''切分一部分数据作为测试集'''
    X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.33, random_state=2017)

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

    '''5折stacking'''
    n_folds = 5
    skf = list(StratifiedKFold(y, n_folds))
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        # print(j, clf)
        dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            # print("Fold", i)
            X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
    # clf = LogisticRegression()
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("blend result")
    print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))


def blending():
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.datasets.samples_generator import make_blobs

    '''创建训练的数据集'''
    data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)

    '''模型融合中使用到的各个单模型'''
    clfs = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]

    '''切分一部分数据作为测试集'''
    X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.33, random_state=2017)

    '''5折stacking'''
    n_folds = 5
    skf = list(StratifiedKFold(y, n_folds))

    '''切分训练数据集为d1,d2两部分'''
    X_d1, X_d2, y_d1, y_d2 = train_test_split(X, y, test_size=0.5, random_state=2017)
    dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
    dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        # print(j, clf)
        '''使用第1个部分作为预测，第2部分来训练模型，获得其预测的输出作为第2部分的新特征。'''
        # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_d1, y_d1)
        y_submission = clf.predict_proba(X_d2)[:, 1]
        dataset_d1[:, j] = y_submission
        '''对于测试集，直接用这k个模型的预测值作为新的特征。'''
        dataset_d2[:, j] = clf.predict_proba(X_predict)[:, 1]
        print("val auc Score: %f" % roc_auc_score(y_predict, dataset_d2[:, j]))

    '''融合使用的模型'''
    # clf = LogisticRegression()
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    clf.fit(dataset_d1, y_d2)
    y_submission = clf.predict_proba(dataset_d2)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("blend result")
    print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))


def lgb():
    import warnings
    warnings.simplefilter('ignore')

    import gc
    import re
    import time

    %matplotlib inline
    import matplotlib.pyplot as plt
    import seaborn as sns

    import numpy as np
    import pandas as pd
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    pd.set_option('float_format', lambda x: '%.6f' % x)

    from IPython.display import display
    pd.options.display.max_rows = None

    from tqdm import tqdm
    tqdm.pandas()

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    import lightgbm as lgb

    # 导入数据

    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')
    df = pd.concat([train, test])
    event = pd.read_csv('../event.csv')
    df = pd.merge(df, event, on='event_id', how='left')

    del train, test, event
    gc.collect()

    # 重命名 columns 区分 event_id 特征
    df.rename(columns={'nhit': 'event_id_nhit',
                       'nhitreal': 'event_id_nhitreal',
                       'energymc': 'event_id_energymc',
                       'thetamc': 'event_id_thetamc',
                       'phimc': 'event_id_phimc',
                       'xcmc': 'event_id_xcmc',
                       'ycmc': 'event_id_ycmc'}, inplace=True)
    df.head()

    # t 统计特征
    df['event_id_t_min'] = df.groupby('event_id')['t'].transform('min')
    df['event_id_t_max'] = df.groupby('event_id')['t'].transform('max')
    df['event_id_t_median'] = df.groupby('event_id')['t'].transform('median')
    df['event_id_t_mean'] = df.groupby('event_id')['t'].transform('mean')

    # t "偏移"时间
    df['t_min_diff'] = df['t'] - df['event_id_t_min']
    df['t_max_diff'] = df['event_id_t_max'] - df['t']
    df['t_median_diff'] = df['event_id_t_median'] - df['t']
    df['t_mean_diff'] = df['event_id_t_mean'] - df['t']

    # 重新排序, 为后面 t 的 rolling 或者 diff 特征做准备
    df = df.sort_values(by=['event_id', 't_min_diff']).reset_index(drop=True)

    # 时间变化特征, 强特
    # 也可以用 rolling 加不同 window_size .std() 来做, 效果比 diff 稍微差一点
    # 试过 rolling + diff 效果比只用一种要差, 我还没搞清楚, 可以多尝试
    for i in [4, 6, 8, 10, 12]:
        df[f't_diff_last_{i}'] = df.groupby('event_id')['t'].diff(periods=i).fillna(0)


    # 位置与中心位置的比例?
    df['x_div_xcmc'] = df['x'] / (df['event_id_xcmc'] + 0.01)
    df['y_div_ycmc'] = df['y'] / (df['event_id_ycmc'] + 0.01)

    # 位置的变化特征, 线上 +3 左右
    for i in range(1, 21):
        df[f'x_diff_last_{i}'] = df.groupby(['event_id'])['x'].diff(periods=i).fillna(0)
        df[f'y_diff_last_{i}'] = df.groupby(['event_id'])['y'].diff(periods=i).fillna(0)

    df['x2'] = df['x'] ** 2
    df['y2'] = df['y'] ** 2
    # 与中心距离的位置变化特征, 线上 +1 左右
    df['dis2c'] = ((df['x'] - df['event_id_xcmc']) ** 2 + (df['y'] - df['event_id_ycmc']) ** 2) ** 0.5
    for i in range(1, 10):
        df[f'dis2c_diff_last_{i}'] = df.groupby(['event_id'])['dis2c'].diff(periods=i).fillna(0)

    # 这个特征是比较有用的 event_id 特征
    df['event_id_realhit_ratio'] = df['event_id_nhitreal'] / df['event_id_nhit']

    # freq encoding
    # 没太大作用, 线上 +0.1
    def freq_enc(df, col):
        vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(vc)
        return df

    df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str')
    df = freq_enc(df, 'terror')
    df = freq_enc(df, 'x_y')

    id_and_label = ['event_id', 'hit_id', 'flag']
    useless_features = [
        'z', 'x_y',
        'event_id_t_mean', 'event_id_t_median', 'event_id_t_min', 'event_id_t_max',
        'event_id_nhit',
    ]
    use_features = [col for col in df.columns if col not in id_and_label + useless_features]

    # 伪标签, 还算有点用, 线上 +0.5
    # 这个规则是观察 train  数据得出
    # t < -900 ==> 0
    # t > 1850 ==> 1
    # q < 0    ==> 1
    test = df[df.flag.isna()]
    df.loc[df.flag.isna() & (df.t < -900), 'flag'] = 0
    df.loc[df.flag.isna() & ((df.t > 1850) | (df.q < 0)), 'flag'] = 1
    train = df[df.flag.notna()]
    train['flag'] = train['flag'].astype('int')

    del df
    gc.collect()

    def run_lgb(df_train, df_test, use_features):
        target = 'flag'
        oof_pred = np.zeros((len(df_train),))
        y_pred = np.zeros((len(df_test),))
        folds = GroupKFold(n_splits=6)  # 6 折比 5 折好一点, 当然有时间有机器可以试下更多的 folds
        for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train[target], train['event_id'])):
            start_time = time.time()
            print(f'Fold {fold + 1}')
            x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
            y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
            train_set = lgb.Dataset(x_train, y_train)
            val_set = lgb.Dataset(x_val, y_val)
            params = {
                'learning_rate': 0.2,
                'metric': 'auc',
                'objective': 'binary',
                'feature_fraction': 0.80,
                'bagging_fraction': 0.75,
                'bagging_freq': 2,
                'n_jobs': -1,
                'seed': 1029,
                'max_depth': 8,
                'num_leaves': 64,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5
            }
            model = lgb.train(params,
                              train_set,
                              num_boost_round=5000,
                              early_stopping_rounds=100,
                              valid_sets=[train_set, val_set],
                              verbose_eval=100)
            oof_pred[val_ind] = model.predict(x_val)
            y_pred += model.predict(df_test[use_features]) / folds.n_splits

            print("Features importance...")
            gain = model.feature_importance('gain')
            feat_imp = pd.DataFrame({'feature': model.feature_name(),
                                     'split': model.feature_importance('split'),
                                     'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
            display(feat_imp)
            used_time = (time.time() - start_time) / 3600
            print(f'used_time: {used_time:.2f} hours')
            del x_train, x_val, y_train, y_val, train_set, val_set
            gc.collect()
        return y_pred, oof_pred

    y_pred, oof_pred = run_lgb(train, test, use_features)

    score = roc_auc_score(train['flag'], oof_pred)
    print('auc: ', score)

    np.save(f'lgb_y_pred_{score}', y_pred)
    np.save(f'lgb_oof_pred_{score}', oof_pred)

    best_threshold = 0.35
    test['flag_pred'] = y_pred
    submission = test[['hit_id', 'flag_pred', 'event_id']]
    submission['flag_pred'] = submission['flag_pred'].apply(lambda x: 1 if x > best_threshold else 0)
    submission = submission.sort_values(by='hit_id')
    submission.to_csv(f'submissions/submission_lgb_{score}_threshold_{best_threshold}.csv',
                      index=False)  # 线上 54.600440509
    submission.flag_pred.value_counts()


def main():
    res = getdata()
    print(res)


if __name__ == "__main__":
    main()
