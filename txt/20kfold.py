from sklearn.model_selection import KFold
import pandas as pd
import lightgbm as lgb

folds = KFold(n_splits=5, shuffle=True, random_state=15)

train = pd.DataFrame()
target = pd.DataFrame()

features = ["", "", ""]
categorical_feats = ['feature_2', 'feature_3']
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature=categorical_feats
                           )
    val_data = lgb.Dataset(train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature=categorical_feats
                           )
