import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import k_means_and_GMM as kg

K = 3
n = 50


def iris_data():
    # 处理数据 拿出数据和标签
    column = ["s_length", "s_width", "p_length", "p_width", "class"]
    data = pd.read_csv("data/iris.data.csv", engine='python', names=column)
    std = StandardScaler()
    X = std.fit_transform(np.array(data[column[0:4]]))
    y = np.array(data[column[4]])
    return X, y


def k_means(X):
    print("k_means:")
    mean, X_flag = kg.k_means(X, K)
    print(mean)
    X = np.hstack((X, X_flag.reshape(-1, 1)))
    kg.cal_accuracy(X, K)
    return mean


def gmm(X):
    print("GMM:")
    X, y_z, mean = kg.gmm(X, K)
    label = np.zeros(X.shape[0])
    for i in range(K):
        label[i * n: (i + 1) * n] = i
    X = np.hstack((X, label.reshape(-1, 1)))
    for i in range(X.shape[0]):
        X[i, -1] = np.argmax(y_z[i, :])
    print(mean)
    kg.cal_accuracy(X, K)


def main():
    X, y = iris_data()
    k_means(X)
    gmm(X)


if __name__ == '__main__':
    main()
