# coding: utf8
"""
@Author: yuhao
@Email: yuhao.hu1992@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def plot_data(X, X_label, data_set, labels):
    color_map = ['red', 'blue', 'purple', 'gray', 'yellow', 'green', 'pink']
    total_labels = labels + [X_label]
    labels_class = list(set(total_labels))
    data_set_label_color = [color_map[labels_class.index(i)] for i in labels]
    X_label_color = color_map[labels_class.index(X_label)]
    plt.figure()
    plt.scatter(data_set[:, 0], data_set[:, 1], c=data_set_label_color)
    plt.scatter(X[0], X[1], c=X_label_color, s=200, marker='*')  # 把预测的点用不同形状和大小标记出来，但是颜色和分类的颜色一致
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def classify(X, data_set, labels, k):
    n_samples = data_set.shape[0]
    diff = np.tile(X, (n_samples, 1)) - data_set
    diff **= 2  # 差值平方
    distance = np.sqrt(diff.sum(axis=1))  # 求欧几里得距离
    sorted_distance_index = distance.argsort()  # 得到按照距离升序排序后，在原始列表里的下标列表
    class_count = {}
    for i in range(k):  # 依次获得k个距离最近的样本
        temp_label = labels[sorted_distance_index[i]]  # 得到样本的标签
        class_count[temp_label] = class_count.get(temp_label, 0) + 1  # 标签相同的进行计数
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)  # 对字典按照值进行降序排序
    return sorted_class_count[0][0]


def test1():
    group, labels = create_data_set()
    X = np.array([0.2, 0.2])
    X_label = classify(X, group, labels, 3)
    print X_label
    plot_data(X, X_label, group, labels)


if __name__ == '__main__':
    test1()
