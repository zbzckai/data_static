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

column_name_1 = 'A5'
column_name_2 = 'A9'
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


column_name = column_name_2 +'_'+column_name_1

data.apply(lambda df:diff_time(df,column_name_1,column_name_2),axis = 1)
