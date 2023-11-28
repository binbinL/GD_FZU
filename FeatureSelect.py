import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

# 编码特征
T_cols = ['employer_type', 'industry', 'work_year']
T_cols_kafang = ['employer_type', 'industry', 'work_year']

# 数值固定不动
static_col = ['year_of_loan', 'house_exist', 'use', 'app_type', 'early_return', 'del_in_18month']

del_col = ['loan_id', 'user_id', 'issue_date', 'earlies_credit_mon', 'title', 'post_code', 'region',
           'policy_code', 'f0', 'f1', 'f2', 'f3', 'f4', 'early_return_amount_3mon', 'censor_status', 'scoring_low',
           'scoring_high', 'known_dero', 'pub_dero_bankrup', 'initial_list_status']
# 连续特征
continuous_col = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                  'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return']

features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
            'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
            'censor_status', 'house_exist', 'work_year', 'use', 'isDefault']
T_cols_new = ['work_year']


# 删除不需要的特征
def delete_col(path, des):
    data = pd.read_csv(path)
    data = data.loc[:, features]
    data.to_csv(des, header=True, index=False)


def deal(path, des):
    data = pd.read_csv(path)
    # 存入字典
    text = {}

    for i in continuous_col:
        tmp = {'min': min(data[i]), 'max': max(data[i])}
        text[i] = tmp
        print(i, '\n min: ', min(data[i]), ' max:', max(data[i]))
        data[i] = round(((data[i] - min(data[i])) / (max(data[i]) - min(data[i]))) * 100, 2)

    with open('./source/transform.json', 'w', encoding='utf-8') as f:
        json.dump(text, f, indent=4, ensure_ascii=False)

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
    data['class'] = data['class'].map(class_list)
    # 类别编码
    for col in T_cols_new:
        lbl = LabelEncoder().fit(data[col])
        data[col] = lbl.transform(data[col])

    data.to_csv(des, header=True, index=False)


if __name__ == '__main__':
    path = 'D:/pythoncode_2/GD/source/train_public.csv'  # 源数据
    des = 'D:/pythoncode_2/GD/source/FeatureSelected.csv'  # 删除不需要特征
    des2 = 'D:/pythoncode_2/GD/source/Dealed.csv'  # 对数据进行编码等处理，生成可进行训练数据

    delete_col(path, des)
    deal(des, des2)
