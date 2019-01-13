# -*- coding: utf-8 -*-
# @Time    : 2019\1\8 0008 7:39
# @Author  : 凯
# @File    : feature_select.py

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import BayesianRidge
feature_num_limit = 200
def get_division_feature_2(data, feature_name):
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns) - 1):
        for j in range(i + 1, len(data[feature_name].columns)):
            new_feature_name.append(data[feature_name].columns[i] + '/' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i]] / data[data[feature_name].columns[j]])

    temp_data = pd.DataFrame(pd.concat(new_feature, axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data, temp_data], axis=1).reset_index(drop=True)
    print(data.shape)
    return data.reset_index(drop=True)


# load data
data = pd.read_csv('middel_data1.csv')
train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
train = train[train['收率'] > 0.87].reset_index(drop=True)
data.fillna(-1, inplace=True)
data.head()
data['样本id']
f_list = data.columns.tolist()
data.shape
use_lesss = ['Unnamed: 0', '样本id']  # 删除非数值特征
for fe in use_lesss:
    f_list.remove(fe)
target = train['收率']
train = data[0:train.shape[0]]
test = data[train.shape[0]:]
train['target'] = target
test.shape
train_data = train[f_list]
train_label = train['target']
train_data.shape
feature_name = f_list

xgb_1 = XGBRegressor(learning_rate =0.005,
        seed=27,
        objective="reg:linear",
        max_depth = 20,
        gamma=0.01,
        subsample=0.8,
        missing= -1,
n_estimators = 5000,
        reg_alpha=0.05,
colsample_bytree=0.8,
        silent=True,
        early_stopping_rounds=200,
        n_jobs=8
)


def get_pic(model, feature_name):
    ans = pd.DataFrame()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)

xgb_1.fit(train_data, train_label)

feature_importance = get_pic(xgb_1,feature_name)
pre_data = xgb_1.predict(train_data)
PRE1 = mean_squared_error(pre_data,target)
df_1 = pd.DataFrame()
df_1['pre_data'] = pre_data
df_1['pre_data'] = target
df_1.to_csv('shujuchakan.csv')
FEATURE_1 = feature_importance

list(feature_importance.iloc[0:40]['name'].values)
data = get_division_feature_2(data,list(feature_importance.iloc[0:20]['name'].values))
data.shape

##再来一次训练去特征值最大的
data.columns
data.fillna(-1, inplace=True)
train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
train = train[train['收率'] > 0.87].reset_index(drop=True)
f_list = data.columns.tolist()
train = train[train['收率'] > 0.87].reset_index(drop=True)
use_lesss = ['Unnamed: 0', '样本id']  # 删除非数值特征
for fe in use_lesss:
    f_list.remove(fe)
target = train['收率']
train = data[0:train.shape[0]]
test = data[train.shape[0]:]
train['target'] = target

train_data = train[f_list]
train_label = train['target']
train_data.shape
test_data = test[f_list]
test_data.shape
feature_name = f_list
xgb_1 = XGBRegressor(learning_rate =0.005,
        seed=27,
        objective="reg:linear",
        max_depth = 20,
        #gamma=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        missing= -1,
        n_estimators = 4000,
early_stopping_rounds=400,
        reg_alpha=0.1,
        n_jobs=3)

def get_pic(model, feature_name):
    ans = pd.DataFrame()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)

xgb_1.fit(train_data, train_label)
feature_importance = get_pic(xgb_1,feature_name)
pre_data = xgb_1.predict(train_data)
PRE2 = mean_squared_error(pre_data,target)
df_2 = pd.DataFrame()
df_2['pre_data'] = pre_data
df_2['target'] = target
df_2.to_csv('shujuchakan2.csv')

test_pre = test[['样本id']]

test_pre['A5'] = xgb_1.predict(test_data)
test_pre.to_csv('sloution.csv')


data.shape
###############################=============================================================================================================交叉验证的数据训练
data = pd.read_csv('middel_data1.csv')
data.fillna(-1, inplace=True)
train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
f_list = data.columns.tolist()
train = train[train['收率'] > 0.87].reset_index(drop=True)
use_lesss = ['Unnamed: 0', '样本id']  # 删除非数值特征
for fe in use_lesss:
    f_list.remove(fe)
target = train['收率']
X_train = data[0:train.shape[0]][f_list]
X_test = data[train.shape[0]:][f_list]
train['target'] = target
import lightgbm as lgb

y_train = target.values

param = {'num_leaves': 120,
         'min_data_in_leaf': 30,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train.loc[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train.loc[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train.loc[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))

len(clf.feature_importance())


def xgb_train(data,train,KFold_num = 5):
    f_list = data.columns.tolist()
    use_lesss = ['Unnamed: 0', '样本id']  # 删除非数值特征
    for fe in use_lesss:
        f_list.remove(fe)
    target = train['target']
    train = data[0:train.shape[0]]
    test = data[train.shape[0]:]
    train['target'] = target
    test.shape
    train_data = train[f_list].reset_index(drop = True)
    train_label = train['target']
    train_data.shape
    feature_name = f_list
    len(feature_name)
    test_data = test[f_list].reset_index(drop = True)
    xgb_params = {'eta': 0.005, 'max_depth': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4,'reg_alpha':0.05}##
    folds = KFold(n_splits=KFold_num, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data, train_label)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(train_data.iloc[trn_idx], train_label[trn_idx])
        val_data = xgb.DMatrix(train_data.iloc[val_idx], train_label[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=100, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(train_data.iloc[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(test_data), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    importance = clf.get_fscore()
    list(importance.keys())
    importance.values()
    importance = pd.DataFrame({'feature': list(importance.keys()), 'importance_values': list(importance.values())})
    importance = importance.sort_values('importance_values', ascending=False)
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))

    return predictions_xgb,importance,oof_xgb,predictions_xgb


predictions_xgb, feature_importance,oof_xgb,predictions_xgb= xgb_train(data,train,5)
data.shape
list(feature_importance.iloc[0:20]['feature'].values)
##data = get_division_feature_2(data,list(feature_importance.iloc[0:20]['feature'].values))


# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

mean_squared_error(target.values, oof_stack)
df_2 = pd.DataFrame()
df_2['pre_data'] = oof_stack
df_2['target'] = target.values
df_2.to_csv('shujuchakan2.csv')



import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
#数据数据为两列数据x和y，有表头
data_pre = pd.read_csv('shujuchakan2.csv')
data_pre['cha'] = data_pre['target'] - data_pre['pre_data']
#通过pandas读取为DataFrame，回归用的是矩阵数据而不是列表，数据为n个样品点和m个特征值，这里特征值只有一个因此换证nx1的矩阵
dataSet_x = data_pre.loc[:, 'target'].as_matrix(columns=None)
#T为矩阵转置把1xn变成nx1
dataSet_x = np.array([dataSet_x]).T
dataSet_y = data_pre.loc[:, 'cha'].as_matrix(columns=None)
dataSet_y = np.array([dataSet_y]).T
#regr为回归过程，fit(x,y)进行回归
data_pre.columns

regr = LinearRegression().fit(dataSet_x, dataSet_y)
data_pre['yc'] = regr.predict(dataSet_x)
data_pre['nyc'] = data_pre['yc'] + data_pre['cha']
mean_squared_error(data_pre['target'], data_pre['pre_data']-data_pre['nyc'])



#通过pandas读取为DataFrame，回归用的是矩阵数据而不是列表，数据为)

solution_1 = pd.read_csv('jinnan_round1_submit_20181227.csv',header=None)
solution_1.columns = ['sample','pre']
solution_1['pre'] = predictions
solution_1.to_csv('solution.csv')

dataSet_x = solution_1.loc[:, 'pre'].as_matrix(columns=None)
#T为矩阵转置把1xn变成nx1
dataSet_x = np.array([dataSet_x]).T
solution_1['nyc'] = regr.predict(dataSet_x)
solution_1['last_solution'] = solution_1['pre'] - solution_1['nyc']
solution_1.to_csv('solution_ii.csv')
#===========================================================================================================================================将特征一个一个的往里面塞，进行特征筛选
# 再一个个往里加，来验证作用
f_list = data.columns.tolist()
use_lesss = ['Unnamed: 0', '样本id']  # 删除非数值特征
for fe in use_lesss:
    f_list.remove(fe)
target = train['收率']
train = data[0:train.shape[0]]
test = data[train.shape[0]:]
train['target'] = target
test.shape
train_data = train[f_list].reset_index(drop=True)
train_label = train['target']
train_data.shape
feature_name = f_list
test_data = test[f_list].reset_index(drop=True)

def find_best_feature(feature_name, cv_fold):
    xgb_model = XGBRegressor(learning_rate=0.005,
                         seed=27,
                         max_depth=20,
                         gamma=0.01,
                         subsample=0.8,
                         n_estimators=20000,
                        #reg_alpha = 0.1,
                         #reg_lambda=2,
                         n_jobs=3)
    dtrain = xgb.DMatrix(train_data[feature_name], label=train_label)
    xgb_param = xgb_model.get_xgb_params()
    bst_xgb = xgb.cv(xgb_param, dtrain, num_boost_round=xgb_model.get_params()['n_estimators'], nfold=cv_fold,
                          metrics='rmse', early_stopping_rounds=200, show_stdv=False,
                          verbose_eval=100)
    m2 = bst_xgb['test-rmse-mean'].values[-1]

    return m2

# 按重要性从前往后
now_feature = []
check = 0
feature_num_limit = 120
feature_importance = list(feature_importance['feature'].values)
for i in range(len(feature_importance)):
    if len(now_feature) >= feature_num_limit: break
    if i == 0:
        now_feature.append(feature_importance[i])
        continue
    now_feature.append(feature_importance[i])
    current_score = find_best_feature(now_feature, 5)
    if current_score > check:
        print('目前特征长度为', len(now_feature), ' 最佳CV', current_score, ' 成功加入第', i + 1, '个', '增值为', current_score - check)
        pd.DataFrame(now_feature, columns=['feature_names']).to_csv( 'new_cross_feature.csv')
        check = current_score
    else:
        print('目前特征长度为', len(now_feature), '进度: ' + str(i), ' 最佳CV', current_score)
        now_feature.pop()


