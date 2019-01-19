# -*- coding: utf-8 -*-
# @Time    : 2019\1\16 0016 7:38
# @Author  : 凯
# @File    : LL2.py

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
import gc

gc.collect()


train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
train.shape
#train = train[[x for x in train.index if x not in vaild.index]].copy()
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
train['收率'] = target
data_2 = pd.concat([train,test],axis=0,ignore_index=True)
data_2 = data_2.fillna(-1)
train.shape
std = np.std(target)
mean = np.mean(target)
mean = 0.01
num = 15
column_mean_std = []
time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7','A20', 'A28', 'B4', 'B9', 'B10', 'B11']
time_column = 'B10'
for time_column in time_columns:
    aa = train.groupby(by = time_column)['收率'].agg({time_column+'_count':'count',time_column+'_mean':'mean',time_column+"_std":'std'}).reset_index()
    aa = aa.fillna(100)
    aa.loc[(aa[time_column+'_std']>std) | (aa[time_column+'_count']<num),time_column+'_mean'] = mean
    aa = aa[[time_column,time_column+'_mean',time_column+"_std"]]
    column_mean_std.append(time_column+'_mean')
    column_mean_std.append(time_column+'_std')
    train = train.merge(aa,on = time_column,how = 'left')
    test = test.merge(aa,on = time_column,how = 'left')
len(column_mean_std)
data_2 = pd.concat([train[column_mean_std],test[column_mean_std]],axis=0,ignore_index=True)
data_2 = data_2.fillna(mean)
train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
train.shape
#train = train[[x for x in train.index if x not in vaild.index]].copy()
# 删除类别唯一的特征
target_col = "收率"

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
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)


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
data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))

categorical_columns = [f for f in data.columns if f not in ['样本id']]
numerical_columns = [f for f in data.columns if f not in categorical_columns]

mean_num = np.mean(target)
mean_num
pred_all = []
tmp = pd.DataFrame()
tmp['样本id'] = train['样本id']
tmp['样本id'] = tmp['样本id'].map(lambda x:x.split('_')[1])
tmp['样本id'] = tmp['样本id'].astype(float)
tmp['target'] = target
tmp.reset_index(drop=True,inplace=True)
len(tmp[tmp['样本id']>2000])
###温度特征data
data['wendu_mean'] = np.mean(data[['A6','A10','A12','A15','A17','A21','A25','A27','B6','B8']],axis=1)
numerical_columns.append('wendu_mean')
# 有风的冬老哥，在群里无意爆出来的特征，让我提升了三个个点，当然也可以顺此继续扩展
data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])

numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')

del data['A1']
del data['A3']
del data['A4']
categorical_columns.remove('A1')
categorical_columns.remove('A3')
categorical_columns.remove('A4')
data['A25'][data['样本id'] == 1590] = data['A25'][data['样本id'] != 1590].value_counts().values[0]
data['A25'] = data['A25'].astype(float)
data = pd.concat([data,data_2],axis=1)
#label encoder
for f in categorical_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
train = data[:train.shape[0]]
test  = data[train.shape[0]:]
#train['target'] = list(target)
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0','intTarget_1.0','intTarget_2.0','intTarget_3.0','intTarget_4.0']
mean_column_cross = []
for f1 in categorical_columns:
    if f1 == 'B14':
        break
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    print(categorical_columns)
    if cate_rate < 0.90:
        for f2 in li:
            col_name = 'B14_'+f1+"_"+f2+'_mean'
            mean_column_cross.append(col_name)
            order_label = train.groupby([f1,'B14'])[f2].agg({'B14_'+f1+"_"+f2+'_mean':'mean'}).reset_index()
            train = train.merge(order_label,on = [f1,'B14',],how = 'left')
            test = test.merge(order_label,on = [f1,'B14'],how = 'left')
            miss_rate = test[col_name].isnull().sum()  / test[col_name].shape[0]
            print(miss_rate)
            if miss_rate  > 0.5:
                train = train.drop([col_name], axis=1)
                test = test.drop([col_name], axis=1)
                mean_column_cross.remove(col_name)
                print('shanchu============',col_name)
train.drop(li+['target'], axis=1, inplace=True)
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

from xgboost.sklearn import XGBClassifier, XGBRegressor

xgb_1 = XGBRegressor(learning_rate=0.005,
                     seed=27,
                     objective="reg:linear",
                     max_depth=40,
                     # gamma=0.01,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     missing=-1,
                     n_estimators=5000,
                     early_stopping_rounds=400,
                     reg_alpha=0.01,
                     n_jobs=8)


def get_pic(model, feature_name):
    ans = pd.DataFrame()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)
numerical_columns +mean_columns + mean_column_cross

train_data = train[mean_columns + numerical_columns +mean_column_cross+categorical_columns]
test_data = test[mean_columns + numerical_columns + mean_column_cross+categorical_columns]
train_data.index
train_label = target.values
len(train_label)
xgb_1.fit(train_data, train_label)
feature_name = train_data.columns.tolist()
feature_importance = get_pic(xgb_1, feature_name)
train_data.shape
####=======================


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
    best_score = modeling_cross_validation(params, train_data[init_cols].values, target.values, nr_folds=5)
    print("初始CV score: {:<8.8f}".format(best_score))

    for f in init_cols:
        best_cols.remove(f)
        score = modeling_cross_validation(params, train_data[best_cols].values, target.values, nr_folds=5)
        diff = best_score - score
        print('-' * 10)
        if diff > 0.0000001:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 移除！！".format(f, score, best_score))
            best_score = score
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f, score, best_score))
            best_cols.append(f)
    print('-' * 10)
    print("优化后CV score: {:<8.8f}".format(best_score))

    return best_cols


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

now_feature = []
check = 100000
feature_num_limit = 200
feature_importance = list(feature_importance['name'].values)
def get_division_feature_2(data, feature_name):
    new_feature = []
    new_feature_name = []
    for i in range(len(data[feature_name].columns) - 1):
        for j in range(i + 1, len(data[feature_name].columns)):
            new_feature_name.append(data[feature_name].columns[i] + '/' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i]] / data[data[feature_name].columns[j]])
            new_feature_name.append(data[feature_name].columns[i] + '*' + data[feature_name].columns[j])
            new_feature.append(data[data[feature_name].columns[i]] * data[data[feature_name].columns[j]])
    temp_data = pd.DataFrame(pd.concat(new_feature, axis=1))
    temp_data.columns = new_feature_name
    data = pd.concat([data, temp_data], axis=1).reset_index(drop=True)
    print(data.shape)
    return data.reset_index(drop=True),new_feature_name

[x for x in feature_importance[0:40] if x not in ['样本id']+categorical_columns]
train_data ,new_feature_name= get_division_feature_2(train_data, [x for x in feature_importance[0:40] if x not in ['样本id']+categorical_columns])
test_data,new_feature_name = get_division_feature_2(test_data, [x for x in feature_importance[0:40] if x not in ['样本id']+categorical_columns])
train_data.shape
train_data.fillna(-1,inplace=True)
test_data.fillna(-1,inplace=True)
xgb_1.fit(train_data, train_label)
feature_name = train_data.columns.tolist()
print('再次训练的时候的特征长度',train_data.shape)
feature_importance = get_pic(xgb_1, feature_name)
feature_importance = list(feature_importance['name'].values)


print('训练完成开始')
file_handle=open('log.txt',mode='a+')
for i in range(len(feature_importance)):
    if len(now_feature) >= feature_num_limit: break
    if i == 0:
        now_feature.append(feature_importance[i])
        continue
    now_feature.append(feature_importance[i])
    params = {'num_leaves': 120,
              'min_data_in_leaf': 30,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.01,
              "min_child_samples": 30,
              "boosting": "gbdt",
              # "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              "metric": 'mse',
              "lambda_l1": 0.1,
              "lambda_l1": 1,
              "verbosity": -1}
    now_feature = now_feature.copy()
    current_score = modeling_cross_validation(params, train_data[now_feature].values, target.values, nr_folds=5)
    if (check - current_score) >0.0000001 :
        print('目前特征长度为', len(now_feature), ' 最佳CV', current_score, ' 成功加入第', i + 1, '个', '增值为', check - current_score)
        file_handle.write('目前特征长度为{}最佳CV{} 成功加入第{}个,加入数据为{} 增值为{}'.format(len(now_feature),current_score, i + 1,feature_importance[i],check - current_score))
        pd.DataFrame(now_feature, columns=['feature_names']).to_csv('new_cross_feature.csv', encoding='gbk')
        check = current_score
    else:
        print('目前特征长度为', len(now_feature), '进度: ' + str(i), ' 最佳CV', current_score)
        now_feature.pop()
file_handle.close()
train_data.columns
new_feature_name_n = [x for x in new_feature_name if x in  now_feature]
numerical_columns_n = [x for x in numerical_columns if x in  now_feature]
mean_columns_n =  [x for x in mean_columns if x in  now_feature]
mean_column_cross_n =  [x for x in mean_column_cross if x in  now_feature]
X_train = train_data[numerical_columns_n +mean_columns_n + mean_column_cross_n + new_feature_name_n].values
X_test = test_data[numerical_columns_n +mean_columns_n + mean_column_cross_n+ new_feature_name_n].values
categorical_columns_n = [x for x in categorical_columns if x in  now_feature]
# one hot
enc = OneHotEncoder()
for f in categorical_columns:
    enc.fit(data[f].values.reshape(-1, 1))
    X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
    X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
print(X_test.shape)
print(X_train.shape)

param = {'num_leaves': 120,
         'min_data_in_leaf': 30,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "lambda_l2": 10,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))
y_train = target.values
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))

##### xgb
xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4,'reg_lambda':100}

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))

# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()
folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10
print(mean_squared_error(y_train, oof_stack))

sub_df = pd.read_csv('./jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = predictions
sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
sub_df.to_csv('solution.csv',encoding='gbk',header=False,index=False)