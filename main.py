import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def compute_cost(theta, X, y):
    # 计算非正则项的代价函数
    inner = np.power((X @ theta.T) - y, 2)
    return np.sum(inner) / (2 * X.shape[0])


def compute_cost_regularization(theta, X, y, lm):
    # 计算正则项的代价函数
    inner = np.power((X @ theta.T) - y, 2) + lm * (theta * theta)
    return np.sum(inner) / (2 * X.shape[0])


def gradient(theta, X, y):
    # 求一个多项式函数的梯度
    temp = (X @ theta.T - y).T @ X / X.shape[0]
    return temp


def dimension(data, dim):
    # 由于是多项式的线性拟合，对数据进行处理
    data = data[None, :]
    data = np.vstack((np.ones_like(data), data)).T
    for i in range(2, dim):
        data = np.insert(data, data.shape[1], np.power(data[:, 1], i), axis=1)
    return data


def armijo_backtrack(theta, X, y, grad, alpha):
    c = 0.3
    now = compute_cost(theta, X, y)
    next_v = compute_cost(theta - alpha * grad, X, y)
    count = 30
    while next_v < now:
        alpha = alpha * 2
        next_v = compute_cost(theta - alpha * grad, X, y)
        count = count - 1
        if count == 0:
            break
    count = 50
    while next_v > now - c * alpha * np.dot(grad, grad.T):
        alpha = alpha / 2
        next_v = compute_cost(theta - alpha * grad, X, y)
        count = count - 1
        if count == 0:
            break
    return alpha


def normal_equ(X, y, max_exp, regularization, lm):
    # 正规方程法，如果regularization为0，则是没有惩罚项的正规矩阵。如果是1，则是带有惩罚项的正规矩阵
    y = y.reshape(len(y), 1)
    X = dimension(X, max_exp)
    theta = np.zeros(max_exp)
    theta = theta.reshape(1, len(theta))

    reg_matrix = np.identity(max_exp)
    reg_matrix[0, 0] = 0
    lm = 1e-4

    if regularization == 1:
        theta = (np.linalg.pinv(X.T @ X - lm * reg_matrix) @ X.T @ y).T  # 正则化
        print(compute_cost_regularization(theta, X, y, lm))
    else:
        theta = (np.linalg.pinv(X.T @ X) @ X.T @ y).T  # 非正则化
        print(compute_cost(theta, X, y))

    X_pred = np.linspace(-5, 5, 24)
    Y_pred = theta @ dimension(X_pred, max_exp).T

    plt.plot(X_pred, Y_pred[0, :])
    plt.plot(X[:, 1], y, 'ro')
    plt.show()


def gradient_descent(X, y, alpha, threshold, max_exp, regularization, lm):
    # 梯度下降法，如果regularization为0，则是无正则项的梯度下降，
    # 如果为1，则为惩罚项系数为lm的带有正则项的梯度下降
    X = dimension(X, max_exp)
    y = y.reshape(len(y), 1)
    theta = np.zeros(max_exp)
    theta = theta[None, :]
    cnt = 0
    count = [cnt]

    if regularization == 1:
        cost_last = compute_cost_regularization(theta, X, y, lm)
        cost = [cost_last]
        while True:
            cnt = cnt + 1
            count.append(cnt)
            print(cost_last)
            grad = gradient(theta, X, y)
            alpha = armijo_backtrack(theta, X, y, grad, alpha)
            theta = (1 - (alpha * lm / X.shape[0])) * theta - alpha * grad  # 正则化
            cost_now = compute_cost_regularization(theta, X, y, lm)
            cost.append(cost_now)
            if cost_last - cost_now < threshold:
                break
            cost_last = cost_now
    else:
        cost_last = compute_cost(theta, X, y)
        cost = [cost_last]
        while True:
            cnt = cnt + 1
            count.append(cnt)
            print(cost_last)
            grad = gradient(theta, X, y)
            alpha = armijo_backtrack(theta, X, y, grad, alpha)
            theta = theta - alpha * grad  # 非正则化
            cost_now = compute_cost(theta, X, y)
            cost.append(cost_now)
            if cost_last - cost_now < threshold:
                break
            cost_last = cost_now

    X_pred = np.linspace(-5, 5, 100)
    Y_pred = theta @ dimension(X_pred, max_exp).T

    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(X_pred, Y_pred[0, :])
    plt.plot(X[:, 1], y, "ro")
    plt.grid(True)

    plt.figure()
    plt.xlabel('iterations')
    plt.ylabel('cost function')
    plt.plot(count, cost)
    plt.show()


def conjugate_gradient(X, y, max_exp, lm):
    # 共轭梯度法
    X = dimension(X, max_exp)
    y = y.reshape((len(y), 1))
    Q = X.T @ X + lm * np.eye(X.shape[1])
    theta = np.zeros((max_exp, 1))
    grad = X.T @ X @ theta - X.T @ y + lm * theta
    r = -grad
    p = r
    for i in range(max_exp):
        pdm = (r.T.dot(r)) / (p.T.dot(Q).dot(p))
        r_prev = r
        theta = theta + pdm * p
        r = r - (pdm * Q).dot(p)
        beta = (r.T.dot(r)) / (r_prev.T.dot(r_prev))
        p = r + beta * p
    ratio = np.poly1d(theta[::-1].reshape(max_exp))
    X_real = np.linspace(-5, 5, 20)
    Y_real = np.sin(X_real) + np.random.randn(X_real.shape[0]) * 0.05
    Y_fit = ratio(X_real)
    plt.plot(X_real, Y_fit, 'b', label='fit_result')
    plt.plot(X_real, Y_real, 'ro', label='real_data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    noise = 0.05
    X = np.linspace(-5, 5, 24)
    y = np.sin(X) + np.random.randn(X.shape[0]) * noise
    alpha = 1e-2
    threshold = 1e-8
    max_exp = 5
    lm = 1e-3
    # gradient_descent(X, y, alpha, threshold, max_exp + 1, 0, lm)  # 梯度下降法
    # normal_equ(X, y, max_exp + 1, 0, lm)                            # 正规方程法

    conjugate_gradient(X, y, max_exp + 1, lm)                       # 共轭梯度法


if __name__ == '__main__':
    main()
