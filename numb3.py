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
data = data.replace(';',':')
data = data.replace('；',':')
data = data.replace('：',':')
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
        data[f+"_tran_num"] = data[f].apply(timeTranSecond)
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
    data[f+"_tran_num"] = data.apply(lambda df: getDuration(df[f]), axis=1)
#===========================================================新增特征
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


def diff_time3(df, column_name_1):
    TIME1 = df[column_name_1]
    try:
        t1, m1 = TIME1.split(":")
        diff_time = (int(t1) * 60 + int(m1))
        return np.ceil(diff_time / 240)
    except:
        diff_time = -1
    return diff_time

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
    del data[cloumns_time_time]
data.head()

time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
for column_name in time_columns:  ##
    data[column_name] = data.apply(lambda df: del_second(df, column_name), axis=1)
time_columns.extend(columns_tmp)
time_columns.sort()
time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A20_1', 'A20_2', 'A24', 'A26', 'A28_1', 'A28_2', 'B4_1', 'B4_2',
                'B5','B7', 'B9_1', 'B9_2', 'B10_1', 'B10_2', 'B11_1', 'B11_2']
data_columns = ['样本id', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17', 'A19', 'A20_1',
                'A20_2', 'A21', 'A22', 'A24', 'A25', 'A26', 'A27', 'A28_1', 'A28_2', 'B1', 'B4_1', 'B4_2', 'B5', 'B6',
                'B7', 'B8', 'B9_1', 'B9_2', 'B10_1', 'B10_2', 'B11_1', 'B11_2', 'B12', 'B14']

data = data.loc[:, data_columns].copy()

###==================================上面为处理数据格式，缺失跟问题数据赋值了NAN======================================
########################################===========================================================================================对数据进行合并，分组组排序，对数据进行合并，每三十合并



##=========================================================生成特征====================================
####=====================================================================================================================特征1 :时间差特征，分类逐次的时间差与阶段性的时间差
for i in range(0, len(time_columns) - 1):
    column_name_1, column_name_2 = time_columns[i], time_columns[i + 1]
    print(column_name_1, column_name_2)
    data[column_name_2 + '_' + column_name_1] = data.apply(lambda df: diff_time(df, column_name_1, column_name_2),
                                                           axis=1)
##A阶段开始的时间到A阶段结束时间A(START,END) A-B(END,ST) B (S-E)
diff_time(data.loc[1525], 'A5', 'A28_2')
data['A28_2' + '_' + 'A5'] = data.apply(lambda df: diff_time(df, 'A5', 'A28_2'), axis=1)
data['B4_1' + '_' + 'A28_2'] = data.apply(lambda df: diff_time(df, 'A28_2', 'B4_1'), axis=1)
data['B11_2' + '_' + 'B4_1'] = data.apply(lambda df: diff_time(df, 'B4_1', 'B11_2'), axis=1)
print(data.shape)
####=====================================================================================================================特征2 :时间截面特征，将时间点进行分组，分为6组

for time_name in time_columns:
    data[time_name] = data.apply(lambda df: diff_time3(df, time_name), axis=1)
data.shape

data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))

categorical_columns = [f for f in data.columns if f not in ['样本id']]
numerical_columns = [f for f in data.columns if f not in categorical_columns]

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