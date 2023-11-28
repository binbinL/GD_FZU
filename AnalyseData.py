import pickle

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def PCA_Process(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    res = pca.transform(X)
    return res


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


def draw(X1, Y1, X2, Y2, X3, Y3, X4, Y4):
    plt.figure(figsize=(15, 6))

    plt.subplot(141)
    plt.title("original data")
    for i in range(500):
        if (Y1[i] == 0):
            plt.scatter(X1[i, 0], X1[i, 1], c="red", marker='.', label='see')
        else:
            plt.scatter(X1[i, 0], X1[i, 1], c="green", marker='.', label='see')
    # plt.legend()

    plt.subplot(142)
    plt.title("oversampling data")
    for i in range(500):
        if (Y2[i] == 0):
            plt.scatter(X2[i, 0], X2[i, 1], c="red", marker='.', label='see')
        else:
            plt.scatter(X2[i, 0], X2[i, 1], c="green", marker='.', label='see')
    # plt.legend()

    plt.subplot(143)
    plt.title("undersampling data")
    for i in range(500):
        if (Y3[i] == 0):
            plt.scatter(X3[i, 0], X3[i, 1], c="red", marker='.', label='see')
        else:
            plt.scatter(X3[i, 0], X3[i, 1], c="green", marker='.', label='see')
    # plt.legend()

    plt.subplot(144)
    plt.title("mixedsampling data")
    for i in range(500):
        if (Y4[i] == 0):
            plt.scatter(X4[i, 0], X4[i, 1], c="red", marker='.', label='see')
        else:
            plt.scatter(X4[i, 0], X4[i, 1], c="green", marker='.', label='see')
    # plt.legend()

    plt.show()


def DataAnalyse():
    data = pd.read_csv('D:/pythoncode_2/GD/source/originaldata.csv')
    x = data.loc[:, :'early_return_amount']
    y = data.loc[:, 'labels']
    pca_x = PCA_Process(x)

    oversampling = pd.read_csv('D:/pythoncode_2/GD/source/oversampling.csv')
    over_x = oversampling.loc[:, :'early_return_amount']
    over_y = oversampling.loc[:, 'labels']
    pca_over_x = PCA_Process(over_x)

    undersampling = pd.read_csv('D:/pythoncode_2/GD/source/undersampling.csv')
    under_x = undersampling.loc[:, :'early_return_amount']
    under_y = undersampling.loc[:, 'labels']
    pca_under_x = PCA_Process(under_x)

    mixedsampling = pd.read_csv('D:/pythoncode_2/GD/source/mixedsampling.csv')
    mixed_x = mixedsampling.loc[:, :'early_return_amount']
    mixed_y = mixedsampling.loc[:, 'labels']
    pca_mixed_x = PCA_Process(mixed_x)

    draw(pca_x, y, pca_over_x, over_y, pca_under_x, under_y, pca_mixed_x, mixed_y)


def countdata(y1, y2):
    plt.figure(figsize=(8, 4))
    barWidth = 0.25
    # 设置柱子的高度
    bars1 = y1
    bars2 = y2
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # 创建柱子
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='训练集', hatch='/')
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='测试集', hatch='\\')

    # 显示数字
    for a, b in zip(r1, y1):  # 柱子上的数字显示
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(r2, y2):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)

    # 添加x轴名称
    plt.xticks([r + barWidth / 2 for r in range(len(bars1))], ['0', '1'])

    plt.xlabel("分类类别")
    plt.ylabel("数量")

    # 创建图例
    plt.legend()
    # 展示图片
    plt.show()


def countsamplingdata(y1, y2, y3):
    plt.figure(figsize=(10, 6))
    barWidth = 0.25
    # 设置柱子的高度
    bars1 = y1
    bars2 = y2
    bars3 = y3
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + 2 * barWidth for x in r1]
    # 创建柱子
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='随机欠采样', hatch='-')
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Smote过采样', hatch='/')
    plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Smote-Tomek混合采样', hatch='\\', color='red')

    # 显示数字
    for a, b in zip(r1, y1):  # 柱子上的数字显示
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(r2, y2):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(r3, y3):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)

    # 添加x轴名称
    plt.xticks([r + barWidth for r in range(len(bars1))], ['0', '1'])

    plt.xlabel("分类类别")
    plt.ylabel("数量")

    # 创建图例
    plt.legend()
    # 展示图片
    plt.show()


# def ResultAnalyse(filename,model):
#     data = pd.read_csv(filename)
#     x = data.loc[:, :'early_return_amount']
#     y = data.loc[:, 'labels']
#
#     theta = grabTree(model)
#     y_hat = np.dot(test_dataMat, theta)  # 得到估计的结果保存到y_hat中
#
#     mark = []
#     for i in range(mtest):
#         res = sigmoid(y_hat[i])
#         if res > 0.5:
#             mark.append(1)
#         else:
#             mark.append(0)
#
#     X = PCA_Process(x)
#     Y_test = y
#
#     plt.figure(figsize=(10, 6))
#     plt.subplot(121)
#     for i in range(len(X)):
#         if (Y_test[i] == 0):
#             plt.scatter(X[i, 0], X[i, 1], c="red", marker='.', label='see')
#         else:
#             plt.scatter(X[i, 0], X[i, 1], c="blue", marker='.', label='see')
#
#     plt.subplot(122)
#     for i in range(len(X)):
#         if (Y_pre[i] == 0):
#             plt.scatter(X[i, 0], X[i, 1], c="red", marker='.', label='see')
#         else:
#             plt.scatter(X[i, 0], X[i, 1], c="green", marker='.', label='see')
#
#     plt.show()
def showfeatures(scores, features):
    plt.figure(figsize=(8, 6))
    barWidth = 0.25
    bars1 = scores
    r1 = np.arange(len(bars1))
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='scores')

    # 添加x轴名称
    plt.xticks([r for r in range(len(bars1))], features, rotation=-270)
    # 创建图例
    plt.legend()
    # 展示图片
    plt.show()


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
if __name__ == '__main__':
    # DataAnalyse()
    #countdata([5526, 1125], [2354, 497])  #

    countsamplingdata([2250, 1125], [5526, 5526], [5499, 5499])

    # features = ['class', 'year_of_loan', 'censor_status', 'house_exist', 'work_year', 'use', 'region',
    #             'industry', 'app_type', 'employer_type', 'policy_code']
    # scores = [53.57, 7.32, 6.80, 6.63, 4.64, 4.54, 2.50, 0.70, 0.19, 0.06, 0.00]
    # showfeatures(scores, features)

    # features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup', 'recircle_u',
    #             'f3', 'initial_list_status', 'del_in_18month', 'total_loan','f4','f2','monthly_payment','f1',
    #             'recircle_b','scoring_high','scoring_low','early_return']
    # print(len(features))
    #
    # scores = [0.189, 0.082, 0.078, 0.067, 0.045, 0.039, 0.030, 0.018, 0.017, 0.010, 0.010,0.006,0.003,0.001,-0.001,-0.024,-0.051,-0.055,-0.347]
    # print(len(scores))
    # showfeatures(scores, features)
    pass
