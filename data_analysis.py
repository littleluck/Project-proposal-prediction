import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

import gc
import re

train = pd.read_csv("./train/train.csv")
resources = pd.read_csv("./resources/resources.csv")
test = pd.read_csv("./test/test.csv")

# print(test.head())

# print(len(train[train["project_is_approved"] == 1]))
# print(len(train[train["project_is_approved"] == 0]))



# print(len(train.columns))
# print(len(test.columns))

# resources.csv
# print(train.shape, test.shape)
resources["resources_total"] = resources["quantity"]*resources["price"]
# groupby之后dfr只有id和resources_total两列，相当于没考虑每个资源的description特征
dfr = resources.groupby(["id"], as_index=False)[["resources_total"]].sum()
# print(dfr.head())
# train中有的id而drf中没有的id，对应的resources_total值填充为-1
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)
# print(test.head())

del dfr
gc.collect()

# 每个项目的每个申请的平均价格
dfr = resources.groupby(["id"], as_index=False)[["resources_total"]].mean()
dfr = dfr.rename(columns={"resources_total":"resources_total_mean"})
train = pd.merge(train, dfr, how="left", on="id").fillna(-1)
test = pd.merge(test, dfr, how="left", on="id").fillna(-1)

del dfr
gc.collect()

# 每个项目申请的资源类别数
dfr = resources.groupby(["id"], as_index=False)[["quantity"]].count()
dfr = dfr.rename(columns={"quantity":"resources_quantity_count"})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

del dfr
gc.collect()

# 每个项目申请的资源总数
dfr = resources.groupby(["id"], as_index=False)[["quantity"]].sum()
dfr = dfr.rename(columns={"quantity":"resources_quantity_sum"})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

del dfr
gc.collect()

# 每个项目的描述
dfr = resources.groupby(["id"], as_index=False)[["description"]].sum()
print("resources:", dfr.head())
train = pd.merge(train, dfr, how='left', on='id').fillna("")
test = pd.merge(test, dfr, how='left', on='id').fillna("")

# print(train.shape, test.shape)

del resources
del dfr
gc.collect()


# 从project_is_approved=1的样本中取出1/5使正负样本均衡
from sklearn.model_selection import train_test_split
# train_true = train[train["project_is_approved"] == 1]   # project_is_approved=1的样本
# train_false = train[train["project_is_approved"] == 0]  # project_is_approved=0的样本
# train_1, train_2 = train_test_split(train_true, test_size=0.2)
# train = pd.concat([train_2, train_false], axis=0).reset_index(drop=True)
# print(len(train))
# print(train.head())

train_true = train[train["project_is_approved"] == 1]   # project_is_approved=1的样本
train_false = train[train["project_is_approved"] == 0]  # project_is_approved=0的样本

train_true, train_true_2 = train_test_split(train_true, test_size=0.2)

del train, train_true
gc.collect()

train = pd.concat([train_true_2, train_false], axis=0).reset_index(drop=True)

del train_true_2, train_false
gc.collect()


# 先处理文本特征，然后连接训练集和测试集，再一起处理其他特征
# 处理特征['project_resource_summary', 'project_title', 'project_essay_1', 
# 'project_essay_2', 'project_essay_3', 'project_essay_4']
from sklearn.feature_extraction.text import TfidfVectorizer

# 由于很多项目没有project_essay_3和project_essay_4，所以需要合并
train_essay1234 = train[train["project_essay_3"] != -1]   # project_essay_3存在的样本
train_essay12 = train[train["project_essay_3"] == -1] # project_essay_3不存在的样本

del train
gc.collect()

# train_essay1234["essay_1"] = train_essay1234.apply(lambda row: ' '.join([str(row['project_essay_1']), 
# str(row['project_essay_2'])]), axis=1)
# train_essay1234["essay_2"] = train_essay1234.apply(lambda row: ' '.join([str(row['project_essay_3']), 
# str(row['project_essay_4'])]), axis=1)

train_essay1234["essay_1"] = train_essay1234["project_essay_1"].astype(str) + train_essay1234["project_essay_2"].astype(str)
train_essay1234["essay_2"] = train_essay1234["project_essay_3"].astype(str) + train_essay1234["project_essay_4"].astype(str)
train_essay12["essay_1"] = train_essay12["project_essay_1"].astype(str)
train_essay12["essay_2"] = train_essay12["project_essay_2"].astype(str)
train = pd.concat([train_essay12, train_essay1234], axis=0).reset_index(drop=True)
# print("train:", train["essay_2"])
train.drop(['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4'], axis=1, inplace=True)

del train_essay12, train_essay1234
gc.collect()

test_essay1234 = test[test["project_essay_3"] != -1]   # project_essay_3存在的样本
test_essay12 = test[test["project_essay_3"] == -1] # project_essay_3不存在的样本

del test
gc.collect()

# test_essay1234["essay_1"] = test_essay1234.apply(lambda row: ' '.join([str(row['project_essay_1']), 
# str(row['project_essay_2'])]), axis=1)
# test_essay1234["essay_2"] = test_essay1234.apply(lambda row: ' '.join([str(row['project_essay_3']), 
# str(row['project_essay_4'])]), axis=1)

test_essay1234["essay_1"] = test_essay1234["project_essay_1"].astype(str) + test_essay1234["project_essay_2"].astype(str)
test_essay1234["essay_2"] = test_essay1234["project_essay_3"].astype(str) + test_essay1234["project_essay_4"].astype(str)
test_essay12["essay_1"] = test_essay12["project_essay_1"].astype(str)
test_essay12["essay_2"] = test_essay12["project_essay_2"].astype(str)
test = pd.concat([test_essay12, test_essay1234], axis=0).reset_index(drop=True)

test.drop(['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4'], axis=1, inplace=True)
del test_essay12, test_essay1234
gc.collect()

# 将4个essay合并到一个
# train["project_essay"] = train["project_essay_1"].astype(str) + train["project_essay_2"].astype(str) + train["project_essay_3"].astype(str) + train["project_essay_4"].astype(str)
# test["project_essay"] = test["project_essay_1"].astype(str) + test["project_essay_2"].astype(str) + test["project_essay_3"].astype(str) + test["project_essay_4"].astype(str)
# print(len(test["project_essay_1"][0]))
# print(len(test["project_essay"][0]))


max_features_ = 100
print(train.shape, test.shape)
for c in ['project_resource_summary', 'project_title', 'essay_1', 'essay_2', 'description']:
    gc.collect()

    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_df=0.9, 
    min_df=3, max_features=max_features_)
    tfidf.fit(train[c].astype(str))
    # print(tfidf.vocabulary_)
    # 对train构建TF-IDF矩阵
    train[c+'_len'] = train[c].map(lambda x:len(str(x)))
    train[c+'_wc'] = train[c].map(lambda x:len(str(x).split(' ')))
    features = pd.DataFrame(tfidf.transform(train[c].astype(str)).toarray())
    features.columns = [c+str(i) for i in range(max_features_)]
    train = pd.concat((train, pd.DataFrame(features)), axis=1, ignore_index=False).reset_index(drop=True)    
    
    del features
    gc.collect()

    # 对test构建TF-IDF矩阵
    test[c+'_len'] = test[c].map(lambda x:len(str(x)))
    test[c+'_wc'] = test[c].map(lambda x:len(str(x).split(' ')))
    features = pd.DataFrame(tfidf.transform(test[c].astype(str)).toarray())
    features.columns = [c+str(i) for i in range(max_features_)]
    test = pd.concat((test, pd.DataFrame(features)), axis=1, ignore_index=False).reset_index(drop=True)    

    del features, tfidf
    gc.collect()

gc.collect()

# print(train.head())
# print(test.head())
# print(train.shape, test.shape)

# g = sns.factorplot(x="teacher_id", y="project_is_approved", data=train)
# g.set_ylabels("project_is_approved Probability")
# plt.show()

print("**********连接训练集和测试集***********")

# 连接train和test
# train_len = len(train)
# dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# del train, test

# 处理project_submitted_datetime特征
import datetime
for dataset in [train, test]:
    dataset["project_submitted_datetime"] = pd.to_datetime(dataset["project_submitted_datetime"])
    # dataset["project_submitted_datetime"+"_question_balance"] = (dataset["project_submitted_datetime"] < datetime.date(2010, 2, 18)).astype(np.int)
    dataset["project_submitted_datetime"+"_year"] = dataset["project_submitted_datetime"].dt.year
    dataset["project_submitted_datetime"+"_quarter"] = dataset["project_submitted_datetime"].dt.quarter
    dataset["project_submitted_datetime"+"_month"] = dataset["project_submitted_datetime"].dt.month
    dataset["project_submitted_datetime"+"_day"] = dataset["project_submitted_datetime"].dt.day
    dataset["project_submitted_datetime"+"_dow"] = dataset["project_submitted_datetime"].dt.dayofweek
    dataset["project_submitted_datetime"+"_wd"] = dataset["project_submitted_datetime"].dt.weekday
    dataset["project_submitted_datetime"+"_hour"] = dataset["project_submitted_datetime"].dt.hour
    dataset["project_submitted_datetime"+"_minute"] = dataset["project_submitted_datetime"].dt.minute


# 对object特征选择one-hot编码或者LabelEncoder编码
train["teacher_prefix"] = train["teacher_prefix"].fillna("Missing") # 发现不管填充什么值，最终都为-1
test["teacher_prefix"] = test["teacher_prefix"].fillna("Missing") # 发现不管填充什么值，最终都为-1
# print(dataset["teacher_id"].value_counts())   # LabelEncoder编码
# print(dataset["teacher_prefix"].isnull().sum())
train["teacher_prefix"] = train["teacher_prefix"].replace(["Teacher", "Dr."], "Rare")
train["teacher_prefix"] = train["teacher_prefix"].replace([-1], "Mrs.")
test["teacher_prefix"] = test["teacher_prefix"].replace(["Teacher", "Dr."], "Rare")
test["teacher_prefix"] = test["teacher_prefix"].replace([-1], "Mrs.")
# print(dataset["teacher_prefix"].value_counts())   # one-hot编码
# print(dataset["school_state"].value_counts())   # school_state取值种类比较多，采用LabelEncoder编码较好
# print(dataset["project_grade_category"].value_counts())   # one-hot编码
# print(dataset["project_subject_categories"].value_counts())   # LabelEncoder编码
# print(dataset["project_subject_subcategories"].value_counts())   # LabelEncoder编码    

# one-hot编码
# train = pd.get_dummies(train, columns=["teacher_prefix"])
# train = pd.get_dummies(train, columns=["project_grade_category"])    
# test = pd.get_dummies(test, columns=["teacher_prefix"])
# test = pd.get_dummies(test, columns=["project_grade_category"])

gc.collect()

# LabelEncoder编码
from sklearn.preprocessing import LabelEncoder
for c in ['teacher_id', 'school_state', 'teacher_prefix', 'project_grade_category']:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].unique()) + list(test[c].unique()))
    train[c] = lbl.fit_transform(train[c])
    test[c] = lbl.fit_transform(test[c])


for c in ['project_subject_categories', 'project_subject_subcategories']:
    for i in range(4):
        lbl = LabelEncoder()
        labels = list(train[c].unique()) + list(test[c].unique())
        labels = [ str(lb).split(',')[i] if len(str(lb).split(',')) > i else '' for lb in labels ]
        lbl.fit(labels)
        train[c+'_'+str(i+1)] = lbl.fit_transform(train[c].map(lambda x: str(x).split(',')[i] if len(str(x).split(',')) > i else '').astype(str))
        test[c+'_'+str(i+1)] = lbl.fit_transform(test[c].map(lambda x: str(x).split(',')[i] if len(str(x).split(',')) > i else '').astype(str))

gc.collect()


# 下面的特征是训练时用不到的
col = ['id','project_resource_summary', 'project_title', 'essay_1', 'essay_2', 'description', 'project_submitted_datetime', 
'project_subject_categories', 'project_subject_subcategories']

# 求train和test共同特征
common_features = [c for c in train.columns if c in test.columns]

# 从共同特征中剔除col中的特征就是最后用到的特征
col = [ c for c in common_features if c not in col ]
col.append("project_is_approved")
# train_y = train["project_is_approved"].astype(int)
train = train[col]  # 训练的数据


# 欠采样
# from imblearn.under_sampling import OneSidedSelection
# from imblearn.under_sampling import TomekLinks
# from imblearn.under_sampling import NearMiss
# from imblearn.combine import SMOTEENN
# # train_y = train["project_is_approved"].astype(int)
# # train.drop(labels=["project_is_approved"], axis=1, inplace=True)
# # oss = OneSidedSelection(random_state=0)
# # nm = NearMiss(ratio='majority', random_state=2018)
# smote = SMOTEENN(random_state=0)
# # tk = TomekLinks(ratio='majority', random_state=2018)
# train_columns = train.columns
# train, train_y = smote.fit_sample(train, train_y)   # One-Sided Selection

# train = pd.DataFrame(data=train, columns=train_columns)

# gc.collect()
# print("欠采样后的train样本数：", train.shape)
# print("test:", test.shape)


# 将train划分为两部分
train_y = train["project_is_approved"].astype(int)
train.drop(labels=["project_is_approved"], axis=1, inplace=True)
train_1, train_2, train_1_y, train_2_y = train_test_split(train, train_y, test_size=0.5, random_state=0)

del train, train_y
gc.collect()


# # 由于正负类样本大约是5:1，所以将正类样本分为5部分，再分别于负类样本一起训练5个分类器，最后将其平均
# train_1, train_2 = train_test_split(train_true, test_size=0.2)  # 五等分
# train_1, train_3 = train_test_split(train_1, test_size=0.25)    # 四等分
# train_1, train_4 = train_test_split(train_1, test_size=0.33)    # 三等分
# train_1, train_5 = train_test_split(train_1, test_size=0.5)     # 二等分
# train1~5分别于train_false连接产生5个训练集


# train_1 = pd.concat([train_1, train_false], axis=0).reset_index(drop=True)
# train_2 = pd.concat([train_2, train_false], axis=0).reset_index(drop=True)
# train_3 = pd.concat([train_3, train_false], axis=0).reset_index(drop=True)
# train_4 = pd.concat([train_4, train_false], axis=0).reset_index(drop=True)
# train_5 = pd.concat([train_5, train_false], axis=0).reset_index(drop=True)


# train_3_y = train_3["project_is_approved"].astype(int)
# train_4_y = train_4["project_is_approved"].astype(int)
# train_5_y = train_5["project_is_approved"].astype(int)

# train_1.drop(labels=["project_is_approved"], axis=1, inplace=True)
# train_2.drop(labels=["project_is_approved"], axis=1, inplace=True)

# train_3.drop(labels=["project_is_approved"], axis=1, inplace=True)
# train_4.drop(labels=["project_is_approved"], axis=1, inplace=True)
# train_5.drop(labels=["project_is_approved"], axis=1, inplace=True)


# 得到预测数据集
# test_id = test["id"]
# # col.remove("project_is_approved") # 删除project_is_approved
# # col.pop()   # 删除最后一个元素
# test = test[col]    # 测试的数据
# test.drop(labels=["project_is_approved"], axis=1, inplace=True)



# 查看每个特征的存储类型
# print(dataset.info())

# 查看对象类型的count,unique,top和freq统计信息
# print(dataset.describe(include=["object"]))
# print(dataset["teacher_id"].isnull().sum())
# print(len(dataset))
# print(len(dataset["teacher_id"].value_counts()))

# 查看dataset中缺失数据
# total = dataset.isnull().sum().sort_values(ascending=False)
# percent = (dataset.isnull().sum()/len(dataset)*100).sort_values(ascending=False)
# missing_dataset = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_dataset.head())
# 缺失数据
# project_essay_4                               251037  96.510005
# project_essay_3                               251037  96.510005
# teacher_prefix                                     5   0.001922

# 查看resources中缺失数据
# total = resources.isnull().sum().sort_values(ascending=False)
# percent = (resources.isnull().sum()/len(resources)*100).sort_values(ascending=False)
# missing_resources = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_resources.head())
# description缺失292条记录

# 开始训练模型并预测
# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression(penalty='l1')
# lr_model.fit(train, train_y)
# prediction = lr_model.predict(test)
# result = pd.DataFrame({"id":test_id, "project_is_approved":prediction.astype(np.int32)})
# result.to_csv("submission.csv", index=False)
# print("预测完成！！！")

# import xgboost as xgb 
# import lightgbm as lgb 
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

# 创建一个字典
params = {}

# 调参
print("开始调参：")
def xgb_evaluate(learning_rate, min_child_weight, colsample_bytree, 
    max_depth, subsample, gamma, reg_alpha):
    params['learning_rate'] = max(learning_rate, 0.01)
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['reg_alpha'] = max(reg_alpha, 0)

    params['n_estimators'] = 100    # 固定基学习器个数

    xgb = XGBClassifier(**params)
    xgb.fit(train_1, train_1_y)
    prediction = xgb.predict(train_2)
    return accuracy_score(train_2_y, prediction)

xgbBO = BayesianOptimization(xgb_evaluate, {
    'learning_rate': (0.01, 0.1),
    'min_child_weight': (1, 20),
    'colsample_bytree': (0.1, 1),
    'max_depth': (5, 15),
    'subsample': (0.5, 1),
    'gamma': (0, 10),
    'reg_alpha': (0, 10),
})

xgbBO.maximize()
print(params)

# xgboost
# model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
# learning_rate=0.05, max_depth=3, min_child_weight=1.7817, 
# n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, 
# subsample=0.5213, silent=1, random_state=7, nthread=-1)

# stacking

# LightGBM
# print("训练开始：")



# lgb_model_1 = LGBMClassifier(boosting_type="gbdt", num_leaves=10, learning_rate=0.05, 
# n_estimators=100, subsample_for_bin=50000, 
# min_child_weight=1, min_child_samples=10, subsample=0.8, colsample_bytree=0.5)
# lgb_model_1.fit(train, train_y)
# # train_3_prediction1 = lgb_model_1.predict(train_3)
# prediction1 = lgb_model_1.predict(test)

# del lgb_model_1
# gc.collect()


# print("XGB:")
# xgb_model_2 = XGBClassifier(n_estimators=100, subsample=0.8, colsample_bytree=0.5)
# xgb_model_2.fit(train, train_y)
# # train_3_prediction2 = xgb_model_2.predict(train_3)
# prediction2 = xgb_model_2.predict(test)

# del xgb_model_2
# gc.collect()

# print("决策树：")
# xgb_model_4 = DecisionTreeClassifier()
# xgb_model_4.fit(train, train_y)
# # train_3_prediction4 = xgb_model_4.predict(train_3)
# prediction4 = xgb_model_4.predict(test)

# del xgb_model_4
# gc.collect()



# print("梯度提升：")
# xgb_model_5 = GradientBoostingClassifier(n_estimators=100, random_state=0, learning_rate=0.1)
# xgb_model_5.fit(train, train_y)
# # train_3_prediction5 = xgb_model_5.predict(train_3)
# prediction5 = xgb_model_5.predict(test)

# del xgb_model_5
# gc.collect()

# print("随机森林：")
# xgb_model_3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features=0.8, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=5,
#             min_weight_fraction_leaf=0.0, n_estimators=100, 
#             oob_score=False, random_state=0, verbose=0, warm_start=False)

# xgb_model_3.fit(train, train_y)
# prediction3 = xgb_model_3.predict(test)

# del xgb_model_3
# gc.collect()


# 硬投票5个预测结果
# import math
# pre_len = len(prediction1)
# prediction = []
# for i in range(0, pre_len):
#     # vote = prediction1[i]+prediction2[i]+prediction3[i]+prediction4[i]+prediction5[i]
#     vote = prediction1[i]+prediction2[i]+prediction4[i]
#     vote = math.floor(vote/2)
#     prediction.append(vote)

# prediction = np.array(prediction)   # 转为数组

# result = pd.DataFrame({"id":test_id, "project_is_approved":prediction.astype(np.int32)})
# result.to_csv("submission.csv", index=False)
# print("预测完成！！！")

# from sklearn.tree import DecisionTreeClassifier
# # Decision Tree
# dt_model = DecisionTreeClassifier()
# dt_model.fit(train, train_y)
# prediction = dt_model.predict(test)
# result = pd.DataFrame({"id":test_id, "project_is_approved":prediction.astype(np.int32)})
# result.to_csv("submission.csv", index=False)
# print("预测完成！！！")


