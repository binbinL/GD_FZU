import pandas as pd


def merge(file1, file2, file3, file4):

    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_csv(file4)

    df_clean = pd.merge(pd.merge(df2, df3, on='loan_id'), df4, on='loan_id')

    df_clean = df_clean.rename(columns={'labels_x': 'lr_predict', 'labels_y': 'cart_predict', 'labels': 'rf_predict'})
    df_clean.insert(loc=1, column='predict', value=0)
    for i in range(len(df_clean)):
        df_clean['predict'][i] = 0 if (df_clean['lr_predict'][i] + df_clean['cart_predict'][i] + df_clean['rf_predict'][
            i]) < 2 else 1

    df_clean.to_csv('D:/pythoncode_2/GD/resultfile/'+file1.split('.')[0]+'_submission.csv', header=True, index=False)


def readfile(filename):
    df = pd.read_csv(filename)
    res = []
    # print(df)
    for i in range(len(df)):
        res.append([df['loan_id'][i], df['lr_predict'][i], df['cart_predict'][i], df['rf_predict'][i], df['predict'][i]])
    return res

def read_csv(file):
    df = pd.read_csv(file)
    cols = list(df.columns)
    mp = dict()
    for i in range(len(df)):
        for col in cols:
            if not mp.get(col, None):
                mp[col] = []
            mp[col].append(df[col][i])
    return mp, len(df)



if __name__ == '__main__':
    read_csv('D:/pythoncode_2/GD/result/lr_over_submission.csv')
    # pre = 'D:/pythoncode_2/GD/Predictsource/'
    # ppre = 'D:/pythoncode_2/GD/result/'
    #
    # merge(pre + 'predict_raw_del.csv', ppre + 'lr_over_submission.csv', ppre + 'CART_mixed_submission.csv',
    #       ppre + 'RF_mixed_submission.csv')

    # readfile(ppre + 'submission2.csv')
