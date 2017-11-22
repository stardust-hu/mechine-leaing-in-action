# coding: utf8
"""
@Author: yuhao
@Email: yuhao.hu1992@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def create_data_set():
    """
    生成测试数据集
    :return: (feature_data, labels, label_name)
            feature_data: 特征矩阵或者数组
            labels: 标签向量或者列表
            feature_name: 各个特征的名字
    """
    feature_data = np.array([[1, 1],
                         [1, 1],
                         [1, 0],
                         [0, 1],
                         [0, 1]])
    labels = ['yes', 'yes', 'no', 'no', 'no']
    feature_name = ['no surfacing', 'flippers']
    return feature_data, labels, feature_name


def entropy(labels):
    """
    计算香农熵
    :param labels: 数据集标签向量或者列表
    :return: ent
    """
    n_sample = len(labels)
    class_counts = pd.value_counts(labels)  # 统计各个类别出现的次数
    class_prob = 1.0 * class_counts / n_sample  # 计算各个类别出现的概率
    ent = -1.0 * class_prob * np.log2(class_prob)
    ent = np.sum(ent)
    return ent


def split_data_set(feature_data, labels, axis, value):
    """
    按照特征值选择数据集
    :param feature_data: 特征数据
    :param labels: 标签数据
    :param axis: 特征数据维度索引
    :param value: 按照特征的某个值删选
    :return: (eturn_feature_data, return_labels)
            eturn_feature_data: 筛选出来的特征, 不包含原特征
            return_labels: 筛选出来标签
    """

    labels = np.array(labels)
    labels = labels.flatten()

    data_filter = feature_data.T[axis] == value

    return_feature_data = feature_data[data_filter]
    return_feature_data = np.delete(return_feature_data, axis, axis=1)

    return_labels = labels[data_filter]

    return return_feature_data, return_labels


def choose_best_feature_split(feature_data, labels):
    """
    计算信息熵增益最大特征索引
    :param feature_data: 特征数据
    :param labels: 标签数据
    :return: baste_feature_axis
    """
    n_feature = feature_data.shape[1]
    base_entropy = entropy(labels)
    baste_info_gain = 0.0
    baste_feature_axis = -1

    for feature_index in range(n_feature):
        unique_val = set(feature_data.T[feature_index])  # 获取该特征下有哪些分类
        new_entropy = 0.0
        for value in unique_val:
            sub_feature_data, sub_labels = split_data_set(feature_data, labels, feature_index, value)
            prob = 1.0 * len(sub_labels) / len(labels)
            new_entropy += prob * entropy(sub_labels)
        info_gain = base_entropy - new_entropy  # 信息熵增益
        if info_gain > baste_info_gain:
            baste_info_gain = info_gain
            baste_feature_axis = feature_index
    return baste_feature_axis


def majority_count(labels):
    """
    投票某个特征下, 出现最多次数的类别, 使用pandas.value_counts实现
    :param labels: 类别列表
    """
    sorted_label_count = pd.value_counts(labels, sort=True, ascending=False)
    return sorted_label_count.index[0]


def create_tree(feature_data, labels, feature_name):
    """
    创建树
    :param feature_data: 特征数据
    :param labels: 标签数据
    :param feature_name: 特征名称
    :return: my_tree
    """

    labels = np.array(labels).flatten()
    feature_name = np.array(feature_name).flatten()

    if len(set(labels)) == 1:  # 当前的数据集中只有一种类别
        return labels[0]
    if feature_data.shape[1] == 0:  # 所有特征分割完毕, 使用投票法选择分类
        return majority_count(labels)

    baste_feature_index = choose_best_feature_split(feature_data, labels)
    baste_feature_name = feature_name[baste_feature_index]
    my_tree = {baste_feature_name: {}}

    feature_name = np.delete(feature_name, baste_feature_index, axis=0)  # 删除最好特征名称
    unique_values = set(feature_data.T[baste_feature_index])
    for value in unique_values:
        sub_feature_name = feature_name[:]
        sub_feature_data, sub_labels = split_data_set(feature_data, labels, baste_feature_index, value)
        my_tree[baste_feature_name][value] = create_tree(sub_feature_data, sub_labels, sub_feature_name)
    return my_tree


def classify(input_tree, sample_data, feature_name):
    """
    预测
    :param input_tree: 训练好的树
    :param sample_data:
    :param feature_name:
    :return: class_label
    """
    sample_data = np.array(sample_data).flatten()
    feature_name = np.array(feature_name).flatten()

    first_str = list(input_tree.keys())[0]
    second_tree = input_tree[first_str]
    feature_index = np.argmax(feature_name==first_str)
    for key in second_tree.keys():
        if sample_data[feature_index] == key:
            if type(second_tree[key]).__name__ == 'dice':
                class_label = classify(second_tree[key], sample_data, feature_name)
            else:
                class_label = second_tree[key]
    return class_label


def classify_data_set(input_tree, feature_data, feature_name):
    """
    预测一批数据
    :param input_tree:
    :param feature_data:
    :param feature_name:
    :return:labels
    """
    labels = []
    n_sample = feature_data.shape[0]
    for si in range(n_sample):
        sample_data = feature_data[si]
        temp_label = classify(input_tree, sample_data, feature_name)
        labels.append(temp_label)
    return labels


def test():
    feature_data, labels, feature_name = create_data_set()

    # print(entropy(labels))

    # print(choose_best_feature_split(feature_data, labels))

    print(create_tree(feature_data, labels, feature_name))


if __name__ == '__main__':
    test()
