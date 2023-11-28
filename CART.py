import pickle

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


# 定义节点类
class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx  # 节点对应的特征索引
        self.threshold = threshold  # 节点对应特征的阈值
        self.value = value  # 如果是叶子节点，则表示预测值
        self.left = left  # 左子树
        self.right = right  # 右子树


# 定义CART树算法
class CARTTree:
    def __init__(self, max_depth=None, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth  # 树的最大深度, 如果为None，则建立尽可能深的树
        self.min_samples_split = min_samples_split  # 决策树分裂所需的最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶子节点所需的最小样本数

    # 计算基尼系数
    def gini(self, y):
        counts = np.bincount(y)  # 计算每个类别的数量
        probs = counts / len(y)  # 计算每个类别的占比
        gini = 1 - np.sum(np.square(probs))  # 计算基尼系数
        return gini

    # 计算平方误差
    def mse(self, y):
        error = np.mean(np.square(y - np.mean(y)))
        return error

    # 计算样本集合的基尼系数或平方误差
    def criterion(self, y):
        if self.task == 'regression':
            return self.mse(y)
        else:
            return self.gini(y)

    # 按照阈值划分数据集
    def split(self, X, y, feature_idx, threshold):
        left_idxs = X[:, feature_idx] <= threshold  # 根据阈值划分左子树
        right_idxs = ~left_idxs  # 根据阈值划分右子树
        return X[left_idxs], X[right_idxs], y[left_idxs], y[right_idxs]

    # 计算对应特征的最佳阈值，最小化划分误差
    def best_split(self, X, y):
        best_feature_idx, best_threshold, best_error = None, None, np.inf
        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])  # 枚举所有可能的阈值
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_idx, threshold)
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                error = self.criterion(y_left) * len(y_left) + self.criterion(y_right) * len(y_right)
                if error < best_error:
                    best_feature_idx, best_threshold, best_error = feature_idx, threshold, error
        return best_feature_idx, best_threshold

    # 建立CART树
    def build_tree(self, X, y, depth=0):
        # 达到最大深度、样本数过少或只有单一类别，则返回叶子节点
        if depth == self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
            # 计算叶子节点的值为所有样本标签的平均值
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        # 寻找最佳的特征和阈值
        best_feature_idx, best_threshold = self.best_split(X, y)
        if best_feature_idx is None:
            # 如果无法找到最佳的特征和阈值，则返回叶子节点
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        # 根据最佳的特征和阈值进行分裂
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature_idx, best_threshold)
        # 分别对左子树和右子树递归调用 build_tree 方法
        left = self.build_tree(X_left, y_left, depth=depth + 1)
        right = self.build_tree(X_right, y_right, depth=depth + 1)
        return TreeNode(feature_idx=best_feature_idx, threshold=best_threshold, left=left, right=right)

    # 预测新的样本
    def predict(self, x):
        node = self.root
        while node.left:  # 如果是非叶子节点，则一直向下遍历
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    # 剪枝函数
    def prune(self, node, X, y):
        # 如果是叶子节点，进行剪枝
        if not node.left and not node.right:
            # 叶子节点的预测值
            leaf_value = np.mean(y)
            # 真实值与叶子节点预测值的差的平方和
            error_leaf = np.sum(np.square(y - leaf_value))
            # 真实值与当前节点预测值的差的平方和
            error_node = np.sum(np.square(y - node.value))
            # 如果剪枝后叶子节点的误差小于剪枝前当前节点的误差，则进行剪枝操作。将当前节点的预测值替换为叶子节点的预测值，并将左右子树置空
            if error_leaf < error_node:
                node.value = leaf_value
                node.left = None
                node.right = None
        else:
            # 如果是非叶子节点，继续向下遍历
            X_left, X_right, y_left, y_right = self.split(X, y, node.feature_idx, node.threshold)
            self.prune(node.left, X_left, y_left)
            self.prune(node.right, X_right, y_right)

    # 训练决策树模型
    def fit(self, X, y):
        self.task = 'regression' if len(np.unique(y)) > 5 else 'classification'
        self.root = self.build_tree(X, y)
        self.prune(self.root, X, y)


def trainModel(filename, model):
    data = pd.read_csv(filename)

    train_df = data.loc[:, features]
    label_df = data.loc[:, 'labels']

    train_df = train_df.values
    label_df = label_df.values

    clf = CARTTree(max_depth=6, min_samples_split=5, min_samples_leaf=3)

    clf.fit(train_df, label_df)
    storeTree(clf, model)


def test(filename, model):
    data = pd.read_csv(filename)

    test_df = data.loc[:, features]
    label_df = data.loc[:, 'labels']

    test_df = test_df.values
    label_df = label_df.values

    clf = grabTree(model)

    predict = np.apply_along_axis(clf.predict, axis=1, arr=test_df)  # 一次性预测所有样本
    predict = np.where(predict < 0.5, 0, 1)

    # print(label_df)
    # print(predict)  # 打印预测结果

    print("=====" * 5)
    print('precision_score: ', metrics.precision_score(label_df, predict))
    print('roc_auc_score: ', metrics.roc_auc_score(label_df, predict))
    print('recall_score: ', metrics.recall_score(label_df, predict))
    print('accuracy_score: ', metrics.accuracy_score(label_df, predict))
    print('f1_score: ', metrics.f1_score(label_df, predict))


def CART_predict(filename, model):
    features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
                'censor_status', 'house_exist', 'work_year', 'use']
    print(filename)
    data = pd.read_csv(filename)

    df = data.loc[:, features]
    datashow = pd.DataFrame(data.loc[:, 'loan_id'])

    df = df.values

    fr = open(model, 'rb')
    clf = pickle.load(fr)

    predict = np.apply_along_axis(clf.predict, axis=1, arr=df)  # 一次性预测所有样本
    predict = np.where(predict < 0.5, 0, 1)

    datashow.insert(1, 'labels', predict)
    datashow.to_csv("./resultfile/" + filename.split('/')[-1].split('.')[0] + "_" + model[8:] + "_submission.csv",
                    index=False, sep=',')



if __name__ == '__main__':
    features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
                'censor_status', 'house_exist', 'work_year', 'use']
    pass
    # trainModel('./source/mixedsampling_ka.csv', './model/CART_mixed_ka')
    # trainModel('./source/oversampling_ka.csv', './model/CART_over_ka')
    # trainModel('./source/undersampling_ka.csv', './model/CART_under_ka')
    #
    # test('./source/test_ka.csv', './model/CART_under_ka')
    # test('./source/test_ka.csv', './model/CART_over_ka')
    # test('./source/test_ka.csv', './model/CART_mixed_ka')

    CART_predict('./PredictSource/predict.csv', './model/CART_mixed_ka')
