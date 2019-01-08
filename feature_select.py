# -*- coding: utf-8 -*-
# @Time    : 2019\1\8 0008 7:39
# @Author  : 凯
# @File    : feature_select.py

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier,XGBRegressor
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
data.fillna(-1, inplace=True)
data.head()
f_list = data.columns.tolist()
use_lesss = ['Unnamed: 0', '样本id', 'A5','A7','A11','A9','A14','A16','A20','A24','A26','A28','B4','B5','B7','B9','B10','B11']  # 删除非数值特征
for fe in use_lesss:
    f_list.remove(fe)
target = train['收率']
train = data[0:train.shape[0]]
test = data[train.shape[0]:]
train['target'] = target

train_data = train[f_list]
train_label = train['target']

feature_name = f_list

xgb_1 = XGBRegressor(learning_rate =0.05,
        seed=27,
        max_depth = 4,
        gamma=0.01,
        subsample=0.8,
        missing= -1,
        n_estimators = 2000,
        reg_lambda = 20,
        n_jobs=8)
def get_pic(model, feature_name):
    ans = pd.DataFrame()
    ans['name'] = feature_name
    ans['score'] = model.feature_importances_
    return ans.sort_values(by=['score'], ascending=False).reset_index(drop=True)

xgb_1.fit(train_data, train_label, verbose=True)
feature_importance = get_pic(xgb_1,feature_name)

# --------------------------------------------------------------------------------------------


# 再一个个往里加，来验证作用
def find_best_feature(feature_name, cv_fold):
    xgb_model = XGBRegressor(learning_rate=0.01,
                         seed=27,
                         max_depth=10,
                         gamma=0.01,
                         subsample=0.8,
                         missing=-1,
                         n_estimators=20000,
                        reg_alpha = 0.1,
                         #reg_lambda=2,
                         n_jobs=8)
    dtrain = xgb.DMatrix(train_data[feature_name], label=train_label)
    xgb_param = xgb_model.get_xgb_params()
    bst_xgb = xgb.cv(xgb_param, dtrain, num_boost_round=xgb_model.get_params()['n_estimators'], nfold=cv_fold,
                          metrics='rmse', early_stopping_rounds=400, show_stdv=False,
                          verbose_eval=100)
    m2 = bst_xgb['test-rmse-mean'].values[-1]
    return m2
find_best_feature(feature_name,5)
# 按重要性从前往后
now_feature = []
check = 0
feature_num_limit = 200
feature_importance = feature_importance['name']
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


