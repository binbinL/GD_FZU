import pandas as pd
import numpy as np
import random
import math
import collections
from joblib import Parallel, delayed
from sklearn import metrics
import pickle


def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(model):
    fr = open(model, 'rb')
    return pickle.load(fr)


class Tree(object):
    """定义一棵决策树"""

    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, dataset):
        """通过递归决策树找到样本所属叶子节点"""
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)


    def describe_tree(self):
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree=None, subsample=0.8, random_state=None):
        """
        随机森林参数
        ----------
        n_estimators:      树数量
        max_depth:         树深度，-1表示不限制深度
        min_samples_split: 节点分裂所需的最小样本数量，小于该值节点终止分裂
        min_samples_leaf:  叶子节点最少样本数量，小于该值叶子被合并
        min_split_gain:    分裂所需的最小增益，小于该值节点终止分裂
        colsample_bytree:  列采样设置，可取[sqrt、log2]。sqrt表示随机选择sqrt(n_features)个特征，
                           log2表示随机选择log(n_features)个特征，设置为其他则不进行列采样
        subsample:         行采样比例
        random_state:      随机种子，设置之后每次生成的n_estimators个样本集不会变
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        """模型训练入口"""
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        # 用于在指定的序列中进行随机抽样，返回指定个数的随机元素。
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # 3种列采样方式
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = int(len(dataset.columns) * 0.7)

        # 并行建立多棵决策树
        '''
        生成器表达式，用于生成多个Delayed对象。
        其中，每一个Delayed对象都会调用函数self._parallel_build_trees(dataset, targets, random_state)，
        其中dataset和targets是随机森林训练数据集和对应的目标变量，而random_state则是一个随机种子，用于控制决策树构建的随机性。
        这个表达式生成的多个Delayed对象会被传递给Parallel函数进行并行化处理，最终生成多个决策树。
        '''
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading") \
            (delayed(self._parallel_build_trees)(dataset, targets, random_state)
             for random_state in random_state_stages)

    # bootstrap有放回抽样生成训练样本集，建立决策树
    def _parallel_build_trees(self, dataset, targets, random_state):

        # 随机抽取列（属性）
        subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
        # 随机抽取一定比例数据
        dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True,
                                       random_state=random_state).reset_index(drop=True)
        # 获得随机抽取的数据（只包含抽取的属性）
        dataset_stage = dataset_stage.loc[:, subcol_index]

        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True,
                                       random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)
        return tree

    def _build_single_tree(self, dataset, targets, depth):
        """递归建立决策树"""
        # 如果该节点的类别全都一样/样本小于分裂所需最小样本数量，则选取出现次数最多的类别。终止分裂
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:

            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth + 1)
                tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth + 1)
                return tree
        # 如果树的深度超过预设值，则终止分裂
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    def choose_best_feature(self, dataset, targets):
        """寻找最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益"""
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # 对可能的分裂阈值求分裂增益 基尼系数最小的
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain

        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets):
        """选择样本中出现次数最多的类别作为叶子节点取值"""
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # 统计每个类别有多少样本，然后计算gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        """根据特征和阈值将样本划分成左右两份，左边小于等于阈值，右边大于阈值"""
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        """输入样本，预测所属类别"""
        res = []
        for _, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，选取出现次数最多的结果作为最终类别
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)


def trainModel(filename, model):
    data = pd.read_csv(filename)

    train_df = data.loc[:, features]
    label_df = data.loc[:, 'labels']

    clf = RandomForestClassifier(n_estimators=7,
                                 max_depth=5,
                                 min_samples_split=7,
                                 min_samples_leaf=2,
                                 min_split_gain=0.001,
                                 colsample_bytree=" ",
                                 subsample=0.7,
                                 random_state=2023)

    clf.fit(train_df, label_df)
    storeTree(clf, model)
    print("over ! ")


def test(filename, model):
    data = pd.read_csv(filename)

    test_df = data.loc[:, features]
    label_df = data.loc[:, 'labels']

    clf = grabTree(model)

    predict = clf.predict(test_df)

    print("=====" * 5)
    # print(label_df)
    # print(predict)
    print('precision_score: ', metrics.precision_score(label_df, predict))
    print('roc_auc_score: ', metrics.roc_auc_score(label_df, predict))
    print('recall_score: ', metrics.recall_score(label_df, predict))
    print('accuracy_score: ', metrics.accuracy_score(label_df, predict))
    print('f1_score: ', metrics.f1_score(label_df, predict))


from functools import partial


def RF_predict(filename, model):
    features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
                'censor_status', 'house_exist', 'work_year', 'use']
    data = pd.read_csv(filename)

    df = data.loc[:, features]
    datashow = pd.DataFrame(data.loc[:, 'loan_id'])

    fr = open(model, 'rb')
    clf = pickle.load(fr)

    predict = clf.predict(df)
    print(type(predict))

    datashow.insert(1, 'labels', predict)
    datashow.to_csv("./resultfile/" + filename.split('/')[-1].split('.')[0] + "_" + model[8:] + "_submission.csv", index=False, sep=',')


if __name__ == '__main__':
    pass
    features = ['interest', 'f0', 'debt_loan_ratio', 'known_outstanding_loan', 'known_dero', 'pub_dero_bankrup',
                'recircle_u', 'recircle_b', 'scoring_high', 'scoring_low', 'early_return', 'class', 'year_of_loan',
                'censor_status', 'house_exist', 'work_year', 'use']
    print(len(features))

    # trainModel('./source/oversampling_ka.csv', './model/RF_over_ka')
    # trainModel('./source/undersampling_ka.csv', './model/RF_under_ka')
    # trainModel('./source/mixedsampling_ka.csv', './model/RF_mixed_ka')
    #
    # test('./source/test_ka.csv', './model/RF_under_ka')
    # test('./source/test_ka.csv', './model/RF_over_ka')
    # test('./source/test_ka.csv', './model/RF_mixed_ka')

    # RF_predict('./PredictSource/predict.csv', './model/RF_mixed_ka')
