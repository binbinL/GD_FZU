import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# 留下贷款id
del_col = ['user_id', 'issue_date', 'earlies_credit_mon', 'title', 'post_code', 'region', 'policy_code', 'f0', 'f1',
           'f2', 'f3', 'f4', 'early_return_amount_3mon', 'censor_status', 'scoring_low', 'scoring_high', 'known_dero',
           'pub_dero_bankrup', 'initial_list_status']

continuous_col = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                  'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return']

features = ['loan_id','interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
            'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
            'censor_status', 'house_exist', 'work_year', 'use']  # 留下load_id
T_cols_new = ['work_year']


def get_raw():
    df = pd.read_csv('D:/pythoncode_2/GD/source/train_public.csv')
    df = df.sample(n=20, random_state=20)
    df = df.drop(columns=['isDefault'], axis=1)
    df.to_csv('D:/pythoncode_2/GD/Predictsource/predict_raw.csv', header=True, index=False)


def get_del(path, des):
    df = pd.read_csv(path)
    # 删除不需要的列
    df = df.loc[:, features]
    # df = df.drop(columns=del_col_kafang)
    df.to_csv(des, header=True, index=False)


def get_transform(path, des):
    df = pd.read_csv(path)
    # 数据标准化
    with open('./source/transform.json', 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    for i in continuous_col:
        df[i] = round(((df[i] - config[i]['min']) / (config[i]['max'] - config[i]['min'])) * 100, 2)
    # for i in continuous_col_kafang:
    #     df[i] = round(((df[i] - config[i]['min']) / (config[i]['max'] - config[i]['min'])) * 100, 2)
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
    for col in T_cols_new:
        lbl = LabelEncoder().fit(df[col])
        df[col] = lbl.transform(df[col])
    df.to_csv(des, header=True, index=False)


if __name__ == '__main__':
    get_raw()
    # path = 'D:/pythoncode_2/GD/Predictsource/predict_raw.csv'
    # des = 'D:/pythoncode_2/GD/PredictSource/predict_undo_transform.csv'
    # des2 = 'D:/pythoncode_2/GD/PredictSource/predict.csv'
    #
    # get_del(path, des)
    # get_transform(des, des2)
