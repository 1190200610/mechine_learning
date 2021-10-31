import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats as st
import pandas as pd


K = 4
INF = 9999999
D = 2
N = 1000
n = 250
colors = "bgrcmykw"


def create_data():
    mean = create_random_num(K, D)
    cov = create_random_cov()
    X = np.zeros((N, D))
    color_index = 0
    for i in range(K):
        x = np.random.multivariate_normal(mean[i], cov=cov[i], size=n)
        plt.scatter(x[:, 0], x[:, 1], c=colors[color_index], marker=".")
        color_index = color_index + 1
        for j in range(n):
            X[j + i * n] = x[j]
    return X


def create_random_cov():
    cov = np.zeros((K, D, D))
    for i in range(K):
        cov[i] = np.identity(D)
        for j in range(D):
            cov[i][j][j] = random.uniform(0.2, 0.4)
    return cov


def create_random_mean(k, f):
    mean = np.zeros((k, f))
    for i in range(k):
        for j in range(f):
            mean[i][j] = random.randint(-5, 5)
    return mean


def create_random_num(k, f):
    num = np.zeros((k, 2))
    for i in range(k):
        for j in range(f):
            num[i][j] = random.uniform(-5, 5)
    return num


def create_random_gmm_num():
    mean = create_random_mean(K, D)
    cov = np.zeros((K, D, D))
    for i in range(K):
        cov[i] = np.identity(D)
    ratio = np.ones(K) / K

    return mean, cov, ratio


def k_means(X, threshold=1e-20):
    X_flag = np.zeros(X.shape[0])
    mean = create_random_num(K, D)
    dis_start = cal_distance_all(X, mean)
    while True:
        dis_end = dis_start
        for i in range(X.shape[0]):
            flag = 0
            min_dis = INF
            for j in range(K):
                if min_dis > cal_distance(X[i, :], mean[j, :]):
                    min_dis = cal_distance(X[i, :], mean[j, :])
                    flag = j
            X_flag[i] = flag
        mean = update_mean(X, X_flag)
        dis_start = cal_distance_all(X, mean)
        if np.fabs(dis_end - dis_start) < threshold:
            break
    print(mean)
    plt.scatter(mean[:, 0], mean[:, 1], c='k', marker='+')
    plt.show()


def gmm(X, threshold=1e-5):
    mean, cov, ratio = create_random_gmm_num()
    last_log_likelihood = cal_log_likelihood(X, mean, cov, ratio)
    iters = 0
    while True:
        y_z = step_e(X, mean, cov, ratio)
        mean, cov, ratio = step_m(X, mean, cov, ratio, y_z)
        now_log_likelihood = cal_log_likelihood(X, mean, cov, ratio)
        print(now_log_likelihood)
        if last_log_likelihood < now_log_likelihood and (now_log_likelihood - last_log_likelihood) < threshold:
            break
        last_log_likelihood = now_log_likelihood
        iters = iters + 1
        if iters >= 20:
            break
    label = np.zeros(X.shape[0])
    for i in range(K):
        label[i * n: (i + 1) * n] = i
    X = np.hstack((X, label.reshape(-1, 1)))
    for i in range(X.shape[0]):
        X[i, -1] = np.argmax(y_z[i, :])
    draw(X, mean)


def draw(X, mean):
    plt.scatter(mean[:, 0], mean[:, 1], marker="+", c="k")
    plt.show()
    colors_array = np.zeros(X.shape[0], dtype=np.str_)
    for i in range(X.shape[0]):
        colors_array[i] = colors[int(X[i, 2])]
    plt.scatter(X[:, 0], X[:, 1], c=colors_array, marker=".")
    plt.show()


def step_e(X, mean, cov, ratio):
    # y_z: 样本混合高斯叠加之后的后验概率
    # ratio: 每种高斯分布所占的比例
    # mean: 每种高斯分布的均值
    # cov: 每种高斯分布的协方差矩阵
    y_z = np.zeros((X.shape[0], K))
    for i in range(X.shape[0]):
        # 计算每个样本在混合高斯之后的后验概率
        ratio_sum = 0
        ratio_pdf = np.zeros(K)
        for j in range(K):
            ratio_pdf[j] = ratio[j] * st.multivariate_normal.pdf(X[i], mean=mean[j], cov=cov[j])
            ratio_sum = ratio_sum + ratio_pdf[j]
        for j in range(K):
            y_z[i, j] = ratio_pdf[j] / ratio_sum
    return y_z


def step_m(X, mean, cov, ratio, y_z):
    new_mean = np.zeros(mean.shape)
    new_cov = np.zeros(cov.shape)
    new_ratio = np.zeros(ratio.shape)
    for j in range(K):
        new_ratio[j] = np.sum(y_z[:, j]) / N
        y = y_z[:, j].reshape(-1, 1)
        new_mean[j, :] = (y.T @ X) / np.sum(y)
        new_cov[j] = ((X - mean[j]).T @ np.multiply((X - mean[j]), y) / np.sum(y))
    return new_mean, new_cov, new_ratio


def update_mean(X, X_flag):
    new_center = np.zeros((K, 2))
    for i in range(K):
        count = 0
        for j in range(X.shape[0]):
            if X_flag[j] == i:
                new_center[i, :] = new_center[i, :] + X[j, :]
                count = count + 1
        if count != 0:
            new_center[i, :] = new_center[i, :] / count
    return new_center


def cal_log_likelihood(X, mean, cov, ratio):
    log_sum = 0
    for i in range(X.shape[0]):
        ratio_pdf_sum = 0
        for j in range(K):
            ratio_pdf_sum = ratio_pdf_sum + ratio[j] * st.multivariate_normal.pdf(X[j], mean=mean[j], cov=cov[j])
        log_sum = log_sum + np.log(ratio_pdf_sum)
    return log_sum


def cal_distance(X, center):
    return np.sum(np.power((X - center), 2))


def cal_distance_all(X, center):
    dis = 0
    for i in range(X.shape[0]):
        for j in range(K):
            dis = dis + cal_distance(X[i, :], center[j, :])
    return dis


def main():
    X = create_data()
    # k_means(X)
    gmm(X)


if __name__ == '__main__':
    main()
