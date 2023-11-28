import numpy as np
import pandas as pd
from sklearn import metrics
import pickle


def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(model):
    fr = open(model, 'rb')
    return pickle.load(fr)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# def sigmoid(x):
#     y = x.copy()  # 对sigmoid函数优化，避免出现极大的数据溢出
#     y[x >= 0] = 1.0 / (1 + np.exp(-x[x >= 0]))
#     y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
#     return y


def gradientDescent(x, y, theta, alpha, m, numIterations):
    """
    :param x: X         m*n
    :param y: Y         m*1
    :param theta:       n*1
    :param alpha: learningRate
    :param m: X 行数
    :param numIterations: 迭代次数
    :return:
    """
    xTrans = x.transpose()  # n*m
    for i in range(0, numIterations):
        hypothesis = sigmoid(np.dot(x, theta))  # sigmoid(m*1)
        loss = hypothesis - y  # m*1
        gradient = np.dot(xTrans, loss) / m  # n*1
        theta = theta - alpha * gradient  # n*1-alpha*n*1
    return theta  # n*1


def trainModel(filename, model):
    traindata = pd.read_csv(filename)
    train_dataMat = traindata.loc[:, features]
    train_labelMat = traindata.loc[:, 'labels']
    # 转成matrix
    train_dataMat = np.mat(train_dataMat.to_numpy())
    train_labelMat = np.mat(train_labelMat.to_numpy())

    mtrain, ntrain = np.shape(train_dataMat)

    numIterations = 100000  # 梯度下降的次数
    alpha = 0.0005  # 每一次的下降步长
    theta = np.ones(shape=(ntrain, 1))  # 参数θ 全1
    theta = gradientDescent(train_dataMat, train_labelMat.transpose(), theta, alpha, mtrain,
                            numIterations)  # 返回训练完毕的参数θ

    storeTree(theta, model)
    print("Model already store")


def LR_predict(filename, model):
    features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
                'censor_status', 'house_exist', 'work_year', 'use']
    data = pd.read_csv(filename)
    dataMat = data.loc[:, features]
    # 存储只保留loan_id
    datashow = pd.DataFrame(data.loc[:, 'loan_id'])
    col = list(data)
    print(col)
    dataMat = dataMat.values.tolist()
    m, n = np.shape(dataMat)

    theta = grabTree(model)
    y_hat = np.dot(dataMat, theta)  # 得到估计的结果保存到y_hat中

    mark = []
    for i in range(m):
        res = sigmoid(y_hat[i])
        if res > 0.5:
            mark.append(1)
        else:
            mark.append(0)
    # dataMat = pd.DataFrame(datashow, columns=col)
    datashow.insert(1, 'labels', mark)

    print("./resultfile/" + filename.split('/')[-1].split('.')[0] + "_" + model[8:] + "_submission.csv")
    datashow.to_csv("./resultfile/" + filename.split('/')[-1].split('.')[0] + "_" + model[8:] + "_submission.csv",
                    index=False, sep=',')


def test(filename, model):
    testdata = pd.read_csv(filename)
    test_dataMat = testdata.loc[:, features]
    test_labelMat = testdata.loc[:, 'labels']
    # 转成list
    test_dataMat = test_dataMat.values.tolist()
    test_labelMat = test_labelMat.values.tolist()

    mtest, ntest = np.shape(test_dataMat)

    theta = grabTree(model)
    y_hat = np.dot(test_dataMat, theta)  # 得到估计的结果保存到y_hat中

    mark = []
    for i in range(mtest):
        res = sigmoid(y_hat[i])
        if res > 0.5:
            mark.append(1)
        else:
            mark.append(0)

    # print('predic result:', mark)
    # print('real result:  ', test_labelMat)

    print("=====" * 5)
    print('precision_score: ', metrics.precision_score(test_labelMat, mark))
    print('roc_auc_score: ', metrics.roc_auc_score(test_labelMat, mark))
    print('recall_score: ', metrics.recall_score(test_labelMat, mark))
    print('accuracy_score: ', metrics.accuracy_score(test_labelMat, mark))
    print('f1_score: ', metrics.f1_score(test_labelMat, mark))



if __name__ == '__main__':
    features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
                'censor_status', 'house_exist', 'work_year', 'use']
    pass

    # trainModel('./source/undersampling_ka.csv', './model/lr_under_ka')
    # trainModel('./source/oversampling_ka.csv', './model/lr_over_ka')
    # trainModel('./source/mixedsampling_ka.csv', './model/lr_mixed_ka')
    #
    # test('./source/test_ka.csv', './model/lr_under_ka')
    # test('./source/test_ka.csv', './model/lr_over_ka')
    # test('./source/test_ka.csv', './model/lr_mixed_ka')

    # LR_predict('./PredictSource/predict.csv', './model/lr_mixed_ka')
