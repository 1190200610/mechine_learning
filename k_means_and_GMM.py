import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats as st
import pandas as pd
from itertools import permutations


K = 4
INF = 9999999
D = 2
N = 1000
n = 250
colors = "bgrcmykw"


def create_data():
    # mean = create_random_mean(K, D)
    mean = create_fixed_mean()
    cov = create_random_cov(K, D)
    X = np.zeros((N, D))
    color_index = 0
    for i in range(K):
        x = np.random.multivariate_normal(mean[i], cov=cov[i], size=n)
        plt.scatter(x[:, 0], x[:, 1], c=colors[color_index], marker=".")
        color_index = color_index + 1
        for j in range(n):
            X[j + i * n] = x[j]
    plt.title("Original Distribution")
    plt.show()
    return X, mean


def create_random_cov(k, d):
    cov = np.zeros((k, d, d))
    for i in range(k):
        cov[i] = np.identity(d)
        for j in range(d):
            cov[i][j][j] = random.uniform(3, 5)
    return cov


def create_random_mean(k, f):
    mean = np.zeros((k, f))
    for i in range(k):
        for j in range(f):
            mean[i][j] = random.randint(-10, 10)
    return mean


def create_fixed_mean():
    # return [[-5, -5], [-5, 5], [5, -5], [5, 5]]
    return [[-4, -4], [-5, 6], [6, -5], [6, 4]]


def create_random_gmm_num(k, d):
    mean = create_random_mean(k, d)
    cov = np.zeros((k, d, d))
    for i in range(k):
        cov[i] = np.identity(d)
    ratio = np.ones(k) / k
    return mean, cov, ratio


def k_means(X, k, threshold=1e-20):
    X_flag = np.zeros(X.shape[0])
    mean = create_random_mean(k, X.shape[1])
    dis_start = cal_distance_all(X, k, mean)
    while True:
        dis_end = dis_start
        for i in range(X.shape[0]):
            flag = 0
            min_dis = INF
            for j in range(k):
                if min_dis > cal_distance(X[i, :], mean[j, :]):
                    min_dis = cal_distance(X[i, :], mean[j, :])
                    flag = j
            X_flag[i] = flag
        mean = update_mean(X, k, X_flag)
        dis_start = cal_distance_all(X, k, mean)
        if np.fabs(dis_end - dis_start) < threshold:
            break
    return mean, X_flag


def gmm(X, k, threshold=1e-5):
    mean, cov, ratio = create_random_gmm_num(k, X.shape[1])
    mean, x = k_means(X, k)
    last_log_likelihood = cal_log_likelihood(X, k, mean, cov, ratio)
    iters = 0
    while True:
        y_z = step_e(X, k, mean, cov, ratio)
        mean, cov, ratio = step_m(X, k, mean, cov, ratio, y_z)
        now_log_likelihood = cal_log_likelihood(X, k, mean, cov, ratio)
        print(now_log_likelihood)
        if last_log_likelihood < now_log_likelihood and (now_log_likelihood - last_log_likelihood) < threshold:
            break
        last_log_likelihood = now_log_likelihood
        iters = iters + 1
        if iters >= 50:
            break
    return X, y_z, mean


def draw_gmm(X, y_z, k, mean):
    label = np.zeros(X.shape[0])
    for i in range(k):
        label[i * n: (i + 1) * n] = i
    X = np.hstack((X, label.reshape(-1, 1)))
    for i in range(X.shape[0]):
        X[i, -1] = np.argmax(y_z[i, :])
    draw(X, k, mean, "GMM ")


def draw_k_means(X, X_flag, k, mean):
    X = np.hstack((X, X_flag.reshape(-1, 1)))
    draw(X, k, mean, "K_means ")


def draw(X, k, real_mean, method):
    for i in range(k):
        x = []
        for j in range(X.shape[0]):
            if X[j, -1] == i:
                x.append(X[j, 0:2])
        x = np.array(x)
        cal_mean = np.mean(x, axis=0)
        min = INF
        real_label = 0
        for p in range(k):
            if cal_distance(cal_mean, real_mean[p]) < min:
                min = cal_distance(cal_mean, real_mean[p])
                real_label = p

        colors_array = np.full(x.shape[0], colors[real_label], dtype=np.str_)
        plt.scatter(x[:, 0], x[:, 1], c=colors_array, marker=".")
    acc = cal_accuracy(X, k)
    plt.title(method + "Classification")
    plt.show()


def cal_accuracy(X, k):
    # 计算准确率
    accuracy = []
    per = np.zeros(k)
    for i in range(k):
        per[i] = i
    per = np.array(list(itertools.permutations([0, 1, 2, 3])))
    for i in range(per.shape[0]):
        count = 0
        for j in range(k):
            for p in range(int(X.shape[0] / k)):
                if X[p + (j * int(X.shape[0] / k)), -1] == per[i][j]:
                    count = count + 1
        accuracy.append(count / X.shape[0])
    num_acc = np.argmax(accuracy)
    acc = accuracy[num_acc]
    print("准确率为："+str(acc))
    return acc


def step_e(X, k, mean, cov, ratio):
    # e步
    # y_z: 样本混合高斯叠加之后的后验概率
    # ratio: 每种高斯分布所占的比例
    # mean: 每种高斯分布的均值
    # cov: 每种高斯分布的协方差矩阵
    y_z = np.zeros((X.shape[0], k))
    for i in range(X.shape[0]):
        # 计算每个样本在混合高斯之后的后验概率
        ratio_sum = 0
        ratio_pdf = np.zeros(k)
        for j in range(k):
            ratio_pdf[j] = ratio[j] * st.multivariate_normal.pdf(X[i], mean=mean[j], cov=cov[j])
            ratio_sum = ratio_sum + ratio_pdf[j]
        for j in range(k):
            y_z[i, j] = ratio_pdf[j] / ratio_sum
    return y_z


def step_m(X, k, mean, cov, ratio, y_z):
    # m步 更新数据
    new_mean = np.zeros(mean.shape)
    new_cov = np.zeros(cov.shape)
    new_ratio = np.zeros(ratio.shape)
    for j in range(k):
        new_ratio[j] = np.sum(y_z[:, j]) / X.shape[0]
        y = y_z[:, j].reshape(-1, 1)
        new_mean[j, :] = (y.T @ X) / np.sum(y)
        new_cov[j] = ((X - mean[j]).T @ np.multiply((X - mean[j]), y) / np.sum(y))
    return new_mean, new_cov, new_ratio


def update_mean(X, k, X_flag):
    new_center = np.zeros((k, X.shape[1]))
    for i in range(k):
        count = 0
        for j in range(X.shape[0]):
            if X_flag[j] == i:
                new_center[i, :] = new_center[i, :] + X[j, :]
                count = count + 1
        if count != 0:
            new_center[i, :] = new_center[i, :] / count
    return new_center


def cal_log_likelihood(X, k, mean, cov, ratio):
    log_sum = 0
    for i in range(X.shape[0]):
        ratio_pdf_sum = 0
        for j in range(k):
            ratio_pdf_sum = ratio_pdf_sum + ratio[j] * st.multivariate_normal.pdf(X[j], mean=mean[j], cov=cov[j])
        log_sum = log_sum + np.log(ratio_pdf_sum)
    return log_sum


def cal_distance(X, center):
    return np.sum(np.power((X - center), 2))


def cal_distance_all(X, k, center):
    dis = 0
    for i in range(X.shape[0]):
        for j in range(k):
            dis = dis + cal_distance(X[i, :], center[j, :])
    return dis


def main():

    # 获取数据 以及真实的均值mean
    X, real_mean = create_data()

    # # k_means
    # mean_k_means, X_flag = k_means(X, K)
    # draw_k_means(X, X_flag, K, real_mean)

    # GMM_EM
    X_gmm, y_z, mean_gmm = gmm(X, K)
    draw_gmm(X, y_z, K, real_mean)


if __name__ == '__main__':
    main()
