import random

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



def Eva_indicators(title, y1, y2, y3):
    plt.figure(figsize=(10, 6))
    barWidth = 0.25
    # 设置柱子的高度
    bars1 = y1
    bars2 = y2
    bars3 = y3
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # 创建柱子
    plt.bar(r1, bars1, width=barWidth, edgecolor='white', label='随机欠采样', hatch='-')
    plt.bar(r2, bars2, width=barWidth, edgecolor='white', label='Smote过采样', hatch='/')
    plt.bar(r3, bars3, width=barWidth, edgecolor='white', label='Smote-Tomek混合采样', hatch='\\', color='red')
    # 添加x轴名称
    plt.xticks([r + barWidth for r in range(len(bars1))],
               ['精确率', 'AUC', '召回率', '准确率', 'F1-Score'])
    #
    # plt.xlabel("评价指标")
    # plt.ylabel("数量")
    plt.title(title)
    # 创建图例
    plt.legend()
    # 展示图片
    plt.show()


def LR_Eva():
    y1 = [0.451, 0.758, 0.698, 0.797, 0.547]
    y2 = [0.394, 0.784, 0.849, 0.743, 0.538]
    y3 = [0.391, 0.778, 0.834, 0.741, 0.532]
    Eva_indicators('Logistic Regression', y1, y2, y3)


def LR_Eva_ka():
    y1 = [0.392, 0.720, 0.656, 0.763, 0.491]
    y2 = [0.327, 0.717, 0.767, 0.684, 0.458]
    y3 = [0.332, 0.724, 0.779, 0.688, 0.465]
    Eva_indicators('逻辑回归', y1, y2, y3)


def CART_Eva():
    y1 = [0.469, 0.754, 0.671, 0.808, 0.552]
    y2 = [0.426, 0.768, 0.752, 0.777, 0.543]
    y3 = [0.426, 0.776, 0.775, 0.777, 0.550]
    Eva_indicators('CART', y1, y2, y3)


def CART_Eva_ka():
    y1 = [0.384, 0.736, 0.716, 0.750, 0.5]
    y2 = [0.339, 0.726, 0.771, 0.698, 0.471]
    y3 = [0.353, 0.736, 0.771, 0.714, 0.484]
    Eva_indicators('CART', y1, y2, y3)


def RF_Eva():
    y1 = [0.473, 0.710, 0.552, 0.813, 0.510]
    y2 = [0.433, 0.808, 0.856, 0.777, 0.575]
    y3 = [0.448, 0.809, 0.839, 0.789, 0.583]
    Eva_indicators('Random Forest', y1, y2, y3)


def RF_Eva_ka():
    y1 = [0.448, 0.684, 0.497, 0.806, 0.471]
    y2 = [0.356, 0.755, 0.825, 0.710, 0.498]
    y3 = [0.367, 0.749, 0.783, 0.727, 0.5]
    Eva_indicators('随机森林', y1, y2, y3)


LR_Eva_ka()
CART_Eva_ka()
RF_Eva_ka()
