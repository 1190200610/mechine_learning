import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets._samples_generator import make_blobs
import scipy.stats as st

K = 4
INF = 9999999
N = 1000
F = 2


def create_data():
    mean = create_random_num(K, F)
    std = create_random_std(K)
    print(mean[1])
    X, y = make_blobs(n_samples=N, n_features=F, centers=mean, cluster_std=std)
    plt.scatter(X[:, 0], X[:, 1], marker='o')
    return X


def create_random_std(k):
    std = np.zeros(k)
    for i in range(k):
        std[i] = random.uniform(0.2, 0.4)
    return std


def create_random_center(k, f):
    mean = np.zeros((k, f))
    for i in range(k):
        for j in range(f):
            mean[i][j] = random.randint(-10, 10)
    return mean


def create_random_num(k, f):
    num = np.zeros((k, 2))
    for i in range(k):
        for j in range(f):
            num[i][j] = random.uniform(-5, 5)
    return num


def create_random_gmm_num():
    mean = create_random_center(K, F)
    cov = np.cov()


def k_means(X, threshold):
    X_flag = np.zeros(N)
    center = create_random_num(K, F)
    dis_start = cal_distance_all(X, center)
    while True:
        dis_end = dis_start
        for i in range(N):
            flag = 0
            min_dis = INF
            for j in range(K):
                if min_dis > cal_distance(X[i, :], center[j, :]):
                    min_dis = cal_distance(X[i, :], center[j, :])
                    flag = j
            X_flag[i] = flag
        center = update_center(X, X_flag)
        dis_start = cal_distance_all(X, center)
        if np.fabs(dis_end - dis_start) < threshold:
            break
    print(center)
    plt.scatter(center[:, 0], center[:, 1], c='r', marker='+')
    plt.show()



def step_e(X, ratio, mean, cov):
    # y_z: 样本混合高斯叠加之后的后验概率
    # ratio: 每种高斯分布所占的比例
    # mean: 每种高斯分布的均值
    # cov: 每种高斯分布的协方差矩阵
    y_z = np.zeros((N, K))
    for i in range(N):
        # 计算每个样本在混合高斯之后的后验概率
        ratio_sum = 0
        ratio_pdf = np.zeros(K)
        for j in range(K):
            ratio_pdf[j] = ratio[j] * st.multivariate_normal.pdf(X[i], mean=mean[j], cov=cov[j])
            ratio_sum = ratio_sum + ratio_pdf[j]
        for j in range(K):
            y_z[i, j] = ratio_pdf[j] / ratio_sum


def step_m(X, ratio, mean, cov, y_z):
    new_mean = np.zeros(mean.shape)
    new_cov = np.zeros(cov)
    new_ratio = np.zeros(ratio)
    for j in range(K):
        new_ratio[j] = np.sum(y_z[:, j]) / N
        y_z = y_z[:, j].reshape(-1, 1)
        new_mean[j, :] = (y_z.T @ X) / np.sum(y_z)
        new_cov[j] = ((X - mean[j]).T @ np.multiply((X - mean[j])), y_z) / np.sum(y_z)
    return new_mean, new_cov, new_ratio


def update_center(X, X_flag):
    new_center = np.zeros((K, 2))
    for i in range(K):
        count = 0
        for j in range(N):
            if X_flag[j] == i:
                new_center[i, :] = new_center[i, :] + X[j, :]
                count = count + 1
        if count != 0:
            new_center[i, :] = new_center[i, :] / count
    return new_center


def cal_distance(X, center):
    return np.sum(np.power((X - center), 2))


def cal_distance_all(X, center):
    dis = 0
    for i in range(N):
        for j in range(K):
            dis = dis + cal_distance(X[i, :], center[j, :])
    return dis


def main():
    X = create_data()
    k_means(X, 1e-30)


if __name__ == '__main__':
    main()
