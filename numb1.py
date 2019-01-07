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
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)

train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding = 'gb18030')

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

# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)

# 删除缺失率超过90%的列
good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]  ##如果为true返回的是比率，na总会排在第一位
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
print(train.columns)
print(test.columns)
del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
#data = data.fillna(-1)
data[0:10]

##日期处理


def diff_time(df,column_name_1,column_name_2):
    TIME1 = df[column_name_1]
    TIME2 = df[column_name_2]
    try:
        t1, m1, s1 = TIME1.split(":")
        t2, m2, s2 = TIME2.split(":")
        diff_time = (int(t2) - int(t1)) * 3600 + (int(m2) - int(m1)) * 60 + (int(s2) - int(s1))
        if diff_time <= 0:
            diff_time = (int(t2) + 24 - int(t1)) * 3600 + (int(m2) - int(m1)) * 60 + (int(s2) - int(s1))
    except:
        diff_time = np.nan
    return diff_time


time_columns = ['A5','A9','A11','A14','A16','A24','A26','B5','B7','A5','A26','A5','B7']
for i in range(0,len(time_columns)-1):
    column_name_1, column_name_2 = time_columns[i],time_columns[i+1]
    print(column_name_1,column_name_2)
    data[column_name_2+'_'+column_name_1] = data.apply(lambda df:diff_time(df,column_name_1,column_name_2),axis = 1)
##查看不同的时间间隔是否会影响后的结果
tmp = data[0:train.shape[0]]
tmp['target'] = target
tmp.groupby(by = column_name_2+'_'+column_name_1)['target'].agg('std')
tmp[column_name_2+'_'+column_name_1].nunique()

sns.stripplot(x=column_name_2+'_'+column_name_1,y='target',data = tmp)
#plt.show()
stats = []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)[:10]

##
stats_df.type.unique()
np.mean(data[stats_df[stats_df.type !='object'].Feature.values]).sort_values()##发现有数据变大
# 些均值相似类似而且在A组中随着系数增加##相似数据为A27,A8,A10,A12,A15,A17,A19
data[stats_df[stats_df.type !='object']]
for i in range(0,stats_df[stats_df.type !='object'].shape[0]-1):
    column_1 = stats_df[stats_df.type !='object']['Feature'].reset_index(drop = True)[i]
    column_2 = stats_df[stats_df.type != 'object']['Feature'].reset_index(drop = True)[i+1]
    data[column_2+'_'+column_1+'continuity'] = data[column_2] - data[column_1]
