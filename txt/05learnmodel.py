# ~)01. 遍历pipeline测试
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn import preprocessing

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, iris.data, iris.target, cv=cv)

pipeline = Pipeline([('vect', CountVectorizer()),
                     ('svc', LinearSVC())])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'svc__loss': ('hinge', 'squared_hinge')
}

# find the best parameters for both the feature extraction and the
# classifier
grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)

# Check that the best model found by grid search is 100% correct on the
# held out evaluation set.
pred = grid_search.fit(train_data, target_train).predict(test_data)
assert_array_equal(pred, target_test)

# ~)02. 调参grid框架
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

# ~)03. 交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import fit_grid_point
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection._validation import _check_is_permutation
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import TimeSeriesSplit

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# ~)04. 保存
import pickle

# 保存到内存中
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(x[0])
# 保存到硬盘中
from sklearn.externals import joblib

joblib.dump(clf, "filename.pkl")


# ~)05. 分层学习
# ~)06. 学习中根据状态换参数函数
