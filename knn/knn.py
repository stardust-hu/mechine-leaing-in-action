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
    """
    生成样本数据

    :return: (group, labels)

               group: 样本特征数据

               labels: 样本对应标签数据
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def plot_data(X, X_label, data_set, labels):
    """
    把预测的点和原始数据集中的点画出来

    :param X: 预测点的特征向量, np.ndarray类型, 大小为(1,  n_feature)一维数据, e.g: array([1, 2])

    :param X_label: 预测点的标签

    :param data_set: 数据集，np.ndarray类型, 大小为(n_samples, n_features), 这里画二维图, n_feature=2

    :param labels: 数据集数据对应标签列表, 数据类型为列表或者是一维的np.ndarray
    """
    color_map = ['red', 'blue', 'purple', 'gray', 'yellow', 'green', 'pink']
    total_labels = labels + [X_label]
    labels_class = list(set(total_labels))
    data_set_label_color = [color_map[labels_class.index(i)] for i in labels]
    X_label_color = color_map[labels_class.index(X_label)]
    plt.figure()
    plt.scatter(data_set[:, 0], data_set[:, 1], c=data_set_label_color)
    plt.scatter(X[0], X[1], c=X_label_color, s=200, marker='*')  # 把预测的点用不同形状和大小标记出来, 但是颜色和分类的颜色一致
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def classify(X, data_set, labels, k):
    """
    计算输入数据所属的类别

    :param X: 预测点的特征向量, np.ndarray类型, 大小为(1,  n_feature)一维数据, e.g: array([1, 2, 1.2, 4.5])

    :param data_set: 数据集，np.ndarray类型, 大小为(n_samples, n_features)

    :param labels: 数据集数据对应标签列表, 数据类型为列表或者是一维的np.ndarray

    :param k: 取和输入数据X距离最近的k个样本

    :return: 列表labels中的某一个标签
    """
    n_samples = data_set.shape[0]
    diff = np.tile(X, (n_samples, 1)) - data_set
    diff **= 2  # 差值平方
    distance = np.sqrt(diff.sum(axis=1))  # 求欧几里得距离
    sorted_distance_index = distance.argsort()  # 得到按照距离升序排序后, 在原始列表里的下标列表
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


def test2():
    """
    使用 sklearn中的digits数据集进行测试
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    imges = digits['images']
    target = digits['target']


if __name__ == '__main__':
    test1()
