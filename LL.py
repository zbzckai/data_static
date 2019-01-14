# -*- coding: utf-8 -*-
# @Time    : 2019\1\14 0014 8:11
# @Author  : 凯
# @File    : LL.py
# -*- coding: utf-8 -*-
# @Time    : 2019\1\11 0011 13:21
# @Author  : 凯
# @File    : numb2.py

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 100)

train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding='gb18030')

stats = []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)[:10]

stats = []
for col in test.columns:
    stats.append((col, test[col].nunique(), test[col].isnull().sum() * 100 / test.shape[0],
                  test[col].value_counts(normalize=True, dropna=False).values[0] * 100, test[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)[:10]

target_col = "收率"

plt.figure(figsize=(8, 6))
plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('yield', fontsize=12)
##plt.show()

# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
# 删除某一类别占比超过90%的列
good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        good_cols.remove(col)
        print(col, rate)
# 暂时不删除，后面构造特征需要
good_cols.append('A1')
good_cols.append('A3')
good_cols.append('A4')

# 删除异常值
train = train[train['收率'] > 0.87]

train = train[good_cols]
good_cols.remove('收率')
test = test[good_cols]

# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.fillna(-1)
data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))
data.shape
train.shape
train = data[:train.shape[0]].copy()
test = data[train.shape[0]:].copy()
test.shape
max(train['样本id'])

mean_all = [0]
std_all = [0]
aa = [[0,0]]
train['target'] = target



for i in range(100):
    try:
        mean_all.append(np.mean(train.loc[((train['样本id']>=(20*(i))) & (train['样本id']<(20*(i+1)))),['target']]).values)
        std_all.append(np.std(train.loc[((train['样本id']>=(20*(i))) & (train['样本id']<(20*(i+1)))),['target']]).values)
        aa.append([20*(i),20*(i+1)])
    except:
        print('不存在数据')
tmp = pd.DataFrame()
tmp['aa'] = aa
tmp['mean_all'] = mean_all
tmp['std_all'] = std_all

train['样本id_cut'] = np.ceil(train['样本id']/20)
test['样本id_cut'] = np.ceil(test['样本id']/20)
tmp = train.groupby(by = '样本id_cut')['target'].mean()
train['yb_cut_mean'] = train['样本id_cut'].map(tmp)
test['yb_cut_mean'] = test['样本id_cut'].map(tmp)
tmp = train.groupby(by = '样本id_cut')['target'].std()
train['yb_cut_std'] = train['样本id_cut'].map(tmp)
test['yb_cut_std'] = test['样本id_cut'].map(tmp)
train.shape
del train['target']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.replace(';',':')
data = data.replace('；',':')
def splt_time(str_1, i):
    try:
        tmp = str_1.split('-')
        if len(tmp) == 2:
            return tmp[i]
        else:
            return np.nan
    except:
        return np.nan


columns_tmp = []
for cloumns_time_time in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    data[cloumns_time_time + '_1'] = data.apply(lambda x: splt_time(x[cloumns_time_time], 0), axis=1)
    data[cloumns_time_time + '_2'] = data.apply(lambda x: splt_time(x[cloumns_time_time], 1), axis=1)
    columns_tmp.append(cloumns_time_time + '_1')
    columns_tmp.append(cloumns_time_time + '_2')
data.head()
data.columns
def del_second(df, column_name):
    try:
        str_tmp = df[column_name]
        t1, m1, s1 = str_tmp.split(":")
        return t1 + ':' + m1
    except:
        return np.nan

def diff_time(df, column_name_1, column_name_2):
    TIME1 = df[column_name_1]
    TIME2 = df[column_name_2]
    try:
        t1, m1 = TIME1.split(":")
        t2, m2 = TIME2.split(":")
        diff_time = ((int(t2) - int(t1)) * 3600 + (int(m2) - int(m1)) * 60)/3600
        if diff_time < 0:
            diff_time = ((int(t2) + 24 - int(t1)) * 3600 + (int(m2) - int(m1)) * 60)/3600
    except:
        diff_time = -1
    return diff_time
data.columns
time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
for column_name in time_columns:  ##
    data[column_name + "_hnew"] = data.apply(lambda df: del_second(df, column_name), axis=1)

time_columns = ['A5_hnew','A7_hnew', 'A9_hnew', 'A11_hnew', 'A14_hnew', 'A16_hnew', 'A20_1', 'A20_2', 'A24_hnew','A26_hnew', 'A28_1', 'A28_2', 'B4_1', 'B4_2',
                'B5_hnew', 'B7_hnew', 'B9_1', 'B9_2', 'B10_1', 'B10_2', 'B11_1', 'B11_2']

for i in range(0, len(time_columns) - 1):
    column_name_1, column_name_2 = time_columns[i], time_columns[i + 1]
    print(column_name_1, column_name_2)
    data[column_name_2 + '_' + column_name_1] = data.apply(lambda df: diff_time(df, column_name_1, column_name_2),
                                                           axis=1)
data.columns
columns_name = list(data.columns)
for i in ['A5_hnew','A7_hnew', 'A9_hnew', 'A11_hnew', 'A14_hnew', 'A16_hnew', 'A20_1', 'A20_2', 'A24_hnew','A26_hnew', 'A28_1', 'A28_2', 'B4_1', 'B4_2',
                'B5_hnew', 'B7_hnew', 'B9_1', 'B9_2', 'B10_1', 'B10_2', 'B11_1', 'B11_2']:
    columns_name.remove(i)
data = data[columns_name]
data.shape
columns_name

###时间乘积特征
data.columns
def cheng_ji(time,column_1):
    try:
        data[time+"_"+column_1] =data[time] *data[column_1]
        return data
    except:
        return data
data = cheng_ji('A7_hnew_A5_hnew','A6')
data = cheng_ji('A9_hnew_A7_hnew','A8')
data = cheng_ji('A11_hnew_A9_hnew','A10')
data = cheng_ji('A14_hnew_A11_hnew','A12')
data = cheng_ji('A14_hnew_A11_hnew','A13')
data = cheng_ji('A16_hnew_A14_hnew','A15')
data = cheng_ji('A20_1_A16_hnew','A17')
data = cheng_ji('A20_1_A16_hnew','A18')
data = cheng_ji('A20_2_A20_1','A17')
data = cheng_ji('A20_2_A20_1','A18')
data = cheng_ji('A24_hnew_A20_2','A21')
data = cheng_ji('A24_hnew_A20_2','A22')
data = cheng_ji('A24_hnew_A20_2','A23')
data = cheng_ji('A26_hnew_A24_hnew','A25')
data = cheng_ji('A28_1_A26_hnew','A27')
data = cheng_ji('A28_2_A28_1','A27')


data = cheng_ji('B4_1_A28_2','A10')
data = cheng_ji('B4_2_B4_1','A12')
data = cheng_ji('B5_hnew_B4_2','A13')
data = cheng_ji('B7_hnew_B5_hnew','A15')
data = cheng_ji('B4_1_A28_2','A10')
data = cheng_ji('B4_2_B4_1','A12')
data = cheng_ji('B5_hnew_B4_2','A13')
data = cheng_ji('B7_hnew_B5_hnew','A15')
data = cheng_ji('B4_1_A28_2','A10')
data = cheng_ji('B4_2_B4_1','A12')
data = cheng_ji('B5_hnew_B4_2','A13')
data = cheng_ji('B7_hnew_B5_hnew','A15')



def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1

    return tm



def timeTranSecond(t):
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0

    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600

    return tm


for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f, '应该在前面被删除了！')


def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1

    return tm


for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
data.head

data.columns

######===================================================================================新增特征

categorical_columns = [f for f in data.columns if f not in ['样本id']]
numerical_columns = [f for f in data.columns if f not in categorical_columns]
# 有风的冬老哥，在群里无意爆出来的特征，让我提升了三个个点，当然也可以顺此继续扩展
data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')
###新加特征
#data.to_csv('all.csv')



del data['A1']
del data['A3']
del data['A4']
categorical_columns.remove('A1')
categorical_columns.remove('A3')
categorical_columns.remove('A4')
# label encoder
for f in categorical_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
train = data[:train.shape[0]]
test = data[train.shape[0]:]
print(train.shape)
print(test.shape)

# train['target'] = list(target)
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
mean_columns = []
for f1 in categorical_columns:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
            mean_columns.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            train[col_name] = train['B14'].map(order_label)
            miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
            if miss_rate > 0:
                train = train.drop([col_name], axis=1)
                mean_columns.remove(col_name)
            else:
                test[col_name] = test['B14'].map(order_label)



train.drop(li + ['target'], axis=1, inplace=True)
print(train.shape)
print(test.shape)

X_train = train[mean_columns + numerical_columns].values
X_train.shape
train.shape
X_test = test[mean_columns + numerical_columns].values
# one hot
enc = OneHotEncoder()
for f in categorical_columns:
    print(f)
    enc.fit(data[f].values.reshape(-1, 1))
    X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')##水平拼接数组
    X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
print(X_train.shape)
print(X_test.shape)

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
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))

##### xgb
xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))

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


sub_df = pd.read_csv('./jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = predictions
sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))


def modeling_cross_validation(params, X, y, nr_folds=5):
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    folds = KFold(n_splits=nr_folds, shuffle=False, random_state=4096)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])

        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=100)
        oof_preds[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)

    score = mean_squared_error(oof_preds, target)

    return score / 2


def featureSelect(init_cols):
    params = {'num_leaves': 120,
              'min_data_in_leaf': 30,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.05,
              "min_child_samples": 30,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              "metric": 'mse',
              "lambda_l1": 0.02,
              "verbosity": -1}
    best_cols = init_cols.copy()
    best_score = modeling_cross_validation(params, train[init_cols].values, target.values, nr_folds=5)
    print("初始CV score: {:<8.8f}".format(best_score))

    for f in init_cols:
        best_cols.remove(f)
        score = modeling_cross_validation(params, train[best_cols].values, target.values, nr_folds=5)
        diff = best_score - score
        print('-' * 10)
        if diff > 0.0000002:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 移除！！".format(f, score, best_score))
            best_score = score
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f, score, best_score))
            best_cols.append(f)
    print('-' * 10)
    print("优化后CV score: {:<8.8f}".format(best_score))

    return best_cols
best_features = featureSelect(train.columns.tolist())
best_features = featureSelect(best_features)
best_features.reverse()
best_features = featureSelect(best_features)
len(best_features)
len(train.columns.tolist())



################
len(train.columns.tolist())

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

data = data[best_features]
data = get_division_feature_2(data,best_features)

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
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))
