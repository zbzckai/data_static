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
data_2.shape