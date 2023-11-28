from io import StringIO
import warnings
import json
import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")  # 忽略告警

col_name = ['loan_id', 'user_id', 'total_loan', 'year_of_loan',
            'interest', 'monthly_payment', 'class', 'employer_type', 'industry',
            'work_year', 'house_exist', 'censor_status', 'issue_date', 'use', 'post_code', 'region', 'debt_loan_ratio',
            'del_in_18month', 'scoring_low', 'scoring_high', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
            'recircle_b', 'recircle_u', 'initial_list_status', 'app_type', 'earlies_credit_mon', 'title', 'policy_code',
            'f0', 'f1', 'f2', 'f3', 'f4', 'early_return', 'early_return_amount', 'early_return_amount_3mon']

f_list = ['total_loan', 'year_of_loan', 'interest', 'monthly_payment', 'class', 'employer_type', 'industry',
          'work_year', 'house_exist', 'use', 'debt_loan_ratio', 'del_in_18month', 'known_outstanding_loan',
          'recircle_b', 'recircle_u', 'app_type', 'early_return', 'early_return_amount']

continuous_col = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                  'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return']

features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
            'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
            'censor_status', 'house_exist', 'work_year', 'use']
T_cols_new = ['work_year']


def original():
    '''

    :return:
    '''
    X_train.insert(len(f_list), 'labels', y_train)
    # shuffle
    originaldata = shuffle(X_train, random_state=2023)
    originaldata.to_csv('D:/pythoncode_2/GD/source/originaldata.csv', header=True, index=False)


def original_ka():
    '''

    :return:
    '''
    X_train.insert(len(features), 'labels', y_train)
    print("original:" ,Counter(y_train))
    # shuffle
    originaldata = shuffle(X_train, random_state=2023)
    originaldata.to_csv('D:/pythoncode_2/GD/source/originaldata_ka.csv', header=True, index=False)


def oversampling():
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smote = SMOTE(random_state=2023)
    X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, y_train)

    print('oversampling: ', Counter(Y_train_balanced))
    # 将labels合并
    X_train_balanced.insert(len(f_list), 'labels', Y_train_balanced)
    # shuffle
    X_train_balanced = shuffle(X_train_balanced, random_state=2023)
    # 保存
    X_train_balanced.to_csv('D:/pythoncode_2/GD/source/oversampling.csv', header=True, index=False)


def oversampling_ka():
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smote = SMOTE(random_state=2023)
    X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, y_train)

    print('oversampling: ', Counter(Y_train_balanced))
    print(X_train_balanced.columns.tolist())
    # 将labels合并
    X_train_balanced.insert(len(features), 'labels', Y_train_balanced)
    # shuffle
    X_train_balanced = shuffle(X_train_balanced, random_state=2023)
    # 保存
    X_train_balanced.to_csv('D:/pythoncode_2/GD/source/oversampling_ka.csv', header=True, index=False)


def undersampling():
    # 将多数类的数量减少为  2 *少数类
    under = RandomUnderSampler(sampling_strategy=0.5)
    X_smote, y_smote = under.fit_resample(X_train, y_train)
    print('undersampling: ', Counter(y_smote))
    X_smote.insert(len(f_list), 'labels', y_smote)
    # shuffle
    X_smote = shuffle(X_smote, random_state=2023)
    X_smote.to_csv('D:/pythoncode_2/GD/source/undersampling.csv', header=True, index=False)


def undersampling_ka():
    # 将多数类的数量减少为  2 *少数类
    under = RandomUnderSampler(sampling_strategy=0.5)
    X_smote, y_smote = under.fit_resample(X_train, y_train)
    print('undersampling: ', Counter(y_smote))
    X_smote.insert(len(features), 'labels', y_smote)
    # shuffle
    X_smote = shuffle(X_smote, random_state=2023)
    X_smote.to_csv('D:/pythoncode_2/GD/source/undersampling_ka.csv', header=True, index=False)


# 混合采样
def smotetomek():
    smotetomek = SMOTETomek(random_state=42)
    X_smote, y_smote = smotetomek.fit_resample(X_train, y_train)
    print('mixedsampling: ', Counter(y_smote))
    X_smote.insert(len(f_list), 'labels', y_smote)
    # shuffle
    X_smote = shuffle(X_smote, random_state=2023)
    X_smote.to_csv('D:/pythoncode_2/GD/source/mixedsampling.csv', header=True, index=False)


def smotetomek_ka():
    smotetomek = SMOTETomek(random_state=42)
    X_smote, y_smote = smotetomek.fit_resample(X_train, y_train)
    print('mixedsampling: ', Counter(y_smote))
    X_smote.insert(len(features), 'labels', y_smote)
    # shuffle
    X_smote = shuffle(X_smote, random_state=2023)
    X_smote.to_csv('D:/pythoncode_2/GD/source/mixedsampling_ka.csv', header=True, index=False)


def testdata():
    X_test.insert(len(f_list), 'labels', y_test)
    X_test.to_csv('D:/pythoncode_2/GD/source/test.csv', header=True, index=False)


def testdata_ka():
    X_test.insert(len(features), 'labels', y_test)
    print("test:",Counter(y_test))
    X_test.to_csv('D:/pythoncode_2/GD/source/test_ka.csv', header=True, index=False)


def analysedata():
    df = pd.read_csv('D:/pythoncode_2/GD/source/originaldata.csv')
    res = df['labels'].value_counts()
    print(res)
    '''
    train_public.csv
    0    8317
    1    1683
    Name: isDefault, dtype: int64
    
    originaldata.csv
    original: Counter({0: 5526, 1: 1125})
    
    oversampling:  Counter({0: 5526, 1: 5526})
    undersampling:  Counter({0: 2250, 1: 1125})
    mixedsampling:  Counter({0: 5499, 1: 5499})
    test: Counter({0: 2354, 1: 497})
    

    '''


if __name__ == '__main__':
    pass
    # analysedata()
    data = pd.read_csv('D:/pythoncode_2/GD/source/Dealed.csv')

    # 删除含有nan的行
    print(data.shape)
    data = data.dropna(axis=0)
    print(data.shape)
    #10000 ->  9508

    x = data.loc[:, features]
    y = data.loc[:, 'isDefault']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2023)
    print(len(X_train))
    print(len(X_test))

    #original_ka()
    # oversampling_ka()
    # undersampling_ka()
    # smotetomek_ka()
    # testdata_ka()
