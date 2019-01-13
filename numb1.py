import numpy as np
import pandas as pd
import warnings
import re
import seaborn as sns
from scipy import sparse
import plotly.offline as py
###from aadahaha import *
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

py.init_notebook_mode(connected=True)
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
test.shape
# 合并数据集
target = train['收率']
print(train.columns)
print(test.columns)
del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.fillna(-1)
data[0:10]


##日期处理
##将秒给去了
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
                'B5',
                'B7', 'B9_1', 'B9_2', 'B10_1', 'B10_2', 'B11_1', 'B11_2']
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
data_tmp = pd.DataFrame()
for time_name in time_columns:
    data_tmp[time_name] = data.apply(lambda df: diff_time3(df, time_name), axis=1)
####=====================================================================================================================特征3 :数值特征的属性
##修改某些异常数据
data['A25'][data['样本id'] == 'sample_1590'] = data['A25'][data['样本id'] != 'sample_1590'].value_counts().values[0]
data_num_values = data[[x for x in data_columns if (x not in time_columns) and (x != '样本id')]]
# 些均值相似类似而且在A组中随着系数增加##相似数据为A27,A8,A10,A12,A15,A17,A19
data_num_values['A25'] = data_num_values['A25'].astype(float)
columns_tmp = data_num_values.columns
for column_tmp_nem  in columns_tmp:
    df = pd.DataFrame(data_num_values[column_tmp_nem].value_counts()).reset_index().sort_values(by='index', ascending=True).reset_index(
    drop=True)
    tmp_index = 0
    tmp_value = 0
    len_num = 0
    new_index = []
    for i in range(0, df.shape[0]):
        len_num += 1
        tmp_index = tmp_index + df.loc[i, 'index']
        tmp_value = tmp_value + df.loc[i, column_tmp_nem]
        if i == (df.shape[0] - 1):
            new_index.extend([new_index[-1]] * (len_num))
            break
        if tmp_value > 20:
            print(tmp_value)
            new_index.extend([df.loc[np.argmax(df.loc[i:i + len_num - 1, column_tmp_nem]), 'index']] * (len_num))
            tmp_index = 0
            tmp_value = 0
            len_num = 0
    df['new_index'] = new_index
    data_num_values[column_tmp_nem] = data_num_values[column_tmp_nem].map(dict(zip(df['index'], df['new_index'])))







for i in range(0, len(columns_tmp) - 1):
    column_1 = columns_tmp[i]
    column_2 = columns_tmp[i + 1]
    data_num_values[column_2 + '_' + column_1 + '_continuity'] = data_num_values[column_2] - data_num_values[column_1]
print(data_num_values.shape)
data_num_values_train = data_num_values[0:train.shape[0]]
data_num_values_train['target'] = target
np.mean(data_num_values_train.target)
np.min(data_num_values_train.target)
np.max(data_num_values_train.target)
for i in data_num_values:
    print('==============================================', i)
    print(data_num_values_train.groupby(by=i).target.agg(['mean', 'count']))


####=====================================================================================================================特征4 增加样本id
a = 'sample_1528'
def sp(df):
    return df.split('_')[1]
data['id_value'] = data['样本id'].apply(lambda df: sp(df))
data['id_value'] = data['id_value'].astype(float)
####=====================================================================================================================特征合并
for col_name_tmp in data_tmp.columns:
    data[col_name_tmp] = data_tmp[col_name_tmp]
for col_name_tmp in data_num_values.columns:
    data[col_name_tmp] = data_num_values[col_name_tmp]
data.shape
data.columns

###
#label encoder
column_name = [x for x in data.columns if x not  in ['样本id','id_value']]

#for f in column_name:
#    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
##
####=====================================================================================================================加入分组目标后各分类标签的均值特征
column_name = [x for x in data.columns if x not  in ['样本id']]
train = data[:train.shape[0]].copy()
test = data[train.shape[0]:].copy()
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
mean_columns = []
for f1 in column_name:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name =  f1 + "_" + f2 + '_mean'
            mean_columns.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            train[col_name] = train[f1].map(order_label)
            miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
            if miss_rate > 0:
                train = train.drop([col_name], axis=1)
                mean_columns.remove(col_name)
            else:
                test[col_name] = test[f1].map(order_label)

train.drop(li + ['target'], axis=1, inplace=True)
print(train.shape)
print(test.shape)
data  = pd.concat([train,test])

column_name = [x for x in data.columns if (x not in ['样本id']) and (x not in [y for y in data.columns if 'mean'  in y ])]
train = data[:train.shape[0]].copy()
test = data[train.shape[0]:].copy()
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
mean_columns = []
for f1 in column_name:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name =  'B14_' + f1 + "_" + f2 + '_mean'
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
data  = pd.concat([train,test])

data.shape
####+====================================================================================================================将一些变量转化为哑声变量
# label encoder
long_col = []
short_col = []
data.fillna(-1)
column_name_1 = [x for x in data.columns if 'mean' not in x ]

for col_name_tmp in column_name_1:
    print(data[col_name_tmp].nunique())
    if data[col_name_tmp].nunique() <= 100:
        tmp = pd.get_dummies(data[col_name_tmp], prefix=col_name_tmp)##, drop_first=True
        short_col.append(col_name_tmp)
        data = pd.concat([data,tmp],axis=1)
    else:
        long_col.append(col_name_tmp)
column_name = list(data.columns)
#for i in short_col:
#    column_name.remove(i)
data = data[column_name]
data.shape
data.to_csv('middel_data1.csv')
data.shape

data.loc[1500:,['样本id','A28_2' + '_' + 'A5']]