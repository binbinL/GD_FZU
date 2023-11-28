import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import json


def getdata():
    path = './source/train_public.csv'
    des = './source/predict_raw.csv'
    df = pd.read_csv(path)
    # 随机抽取50行
    df = df.sample(n=500)
    # 删除某列
    df = df.drop('isDefault', axis=1)
    df.to_csv(des, header=True, index=False)


def Ka():
    path = './source/train_public.csv'
    df = pd.read_csv(path)

    # 编码特征
    T_cols = ['employer_type', 'industry', 'work_year']
    # 类别编码
    class_list = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
    }
    df['class'] = df['class'].map(class_list)
    # 类别编码
    for col in T_cols:
        lbl = LabelEncoder().fit(df[col])
        df[col] = lbl.transform(df[col])

    features = ['employer_type', 'industry', 'work_year', 'class', 'year_of_loan', 'censor_status', 'house_exist',
                'use', 'app_type', 'policy_code', 'region']

    print(len(df))
    df = df.dropna(axis=0)
    print(len(df))

    X = df.loc[:, features].values
    y = df.loc[:, 'isDefault'].values

    from sklearn.feature_selection import SelectKBest

    selector = SelectKBest(chi2, k=len(features))
    selector.fit(X, y)

    scores = -np.log10(selector.pvalues_)
    print(scores)

    indices = np.argsort(scores)[::-1]

    for i in range(len(scores)):
        print("{:.2f} {}".format(scores[indices[i]], features[indices[i]]))


'''
53.57 class
7.32 year_of_loan
6.80 censor_status
6.63 house_exist
4.64 work_year
4.54 use
2.50 region
0.70 industry
0.19 app_type
0.06 employer_type
0.00 policy_code

'''


def perx():
    path = './source/train_public.csv'
    df = pd.read_csv(path)

    # 编码特征
    T_cols = ['employer_type', 'industry', 'work_year']
    # 类别编码
    class_list = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
    }
    df['class'] = df['class'].map(class_list)
    # 类别编码
    for col in T_cols:
        lbl = LabelEncoder().fit(df[col])
        df[col] = lbl.transform(df[col])

    del_col = ['loan_id', 'user_id', 'issue_date', 'class', 'year_of_loan', 'censor_status', 'house_exist', 'work_year',
               'use', 'industry', 'app_type', 'employer_type',
               'policy_code', 'region', 'post_code', 'title', 'early_return_amount_3mon', 'earlies_credit_mon']

    df = df.drop(columns=del_col)

    features = df.columns.tolist()
    # features = features[:-1]
    print(features)

    print(len(df))
    df = df.dropna(axis=0)
    print(len(df))

    X = df.loc[:, :'early_return']
    print(len(X.columns))
    y = df.loc[:, 'isDefault']

    # 计算每个特征与预测值之间的皮尔逊系数
    corr_values = X.corrwith(y)
    print(len(corr_values))

    # 将结果存储在一个DataFrame中
    result_df = pd.DataFrame({'feature': X.columns, 'corr_value': corr_values})

    # 按照皮尔逊系数从大到小排序
    result_df = result_df.sort_values(by='corr_value', ascending=False)

    print(result_df)
    '''
interest                              interest    0.189899
f0                                          f0    0.082122
debt_loan_ratio                debt_loan_ratio    0.077645
known_outstanding_loan  known_outstanding_loan    0.066881
known_dero                          known_dero    0.045279
pub_dero_bankrup              pub_dero_bankrup    0.039578
recircle_u                          recircle_u    0.029797

f3                                          f3    0.018399
initial_list_status        initial_list_status    0.016770
del_in_18month                  del_in_18month    0.009953
total_loan                          total_loan    0.009721
f4                                          f4    0.005845
f2                                          f2    0.002637
monthly_payment                monthly_payment    0.001214
f1                                          f1   -0.001409

recircle_b                          recircle_b   -0.024355
scoring_high                      scoring_high   -0.050748
scoring_low                        scoring_low   -0.055300
early_return                      early_return   -0.346637


    '''


features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
            'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
            'censor_status', 'house_exist', 'work_year', 'use']

if __name__ == '__main__':
    pass
    #Ka()
    #
    # perx()
