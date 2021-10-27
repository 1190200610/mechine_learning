import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("WebAgg")


def create_data_naive_bayesian(m, is_naive):
    positive_mean = [0.5, 1]
    positive_rate = 0.5
    negative_mean = [-0.5, -1]
    negative_rate = 1 - positive_rate
    X = np.zeros((m, 2))
    y = np.zeros(m)
    cov_11 = 0.3
    cov_22 = 0.2
    if is_naive == 1:
        cov_12 = cov_21 = 0
    else:
        cov_12 = cov_21 = 0.1
    cov = [[cov_11, cov_12], [cov_21, cov_22]]      # X的两个维度的协方差矩阵
    positive_num = np.ceil(positive_rate * m).astype(np.int32)
    negative_num = np.ceil(negative_rate * m).astype(np.int32)
    X[:positive_num, :] = np.random.multivariate_normal(positive_mean, cov, size=positive_num)      # 根据协方差矩阵生成正态分布
    X[positive_num:, :] = np.random.multivariate_normal(negative_mean, cov, size=negative_num)      # 将正类反类区分开
    y[:positive_num] = 1
    y[positive_num:] = 0
    plt.scatter(X[:positive_num, 0], X[:positive_num, 1], c='r')
    plt.scatter(X[positive_num:, 0], X[positive_num:, 1], c='g')

    return X, y


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def hypothesis(theta, X):
    return sigmoid(X @ theta.T)


def cost_function(X, y, theta):
    log_likelihood = np.sum(y.reshape(1, -1) @ np.log(hypothesis(theta, X)))
    return -(log_likelihood / X.shape[0])


def cost_function_reg(X, y, theta, lm):
    log_likelihood = np.sum(y.reshape(1, -1) @ np.log(hypothesis(theta, X)))
    reg = np.sum(theta @ theta.T)
    log_likelihood = log_likelihood - (lm / 2) * reg
    return -(log_likelihood / X.shape[0])


def gradient(X, y, theta):
    return (hypothesis(theta, X).reshape(-1, 1) - y.reshape(-1, 1)).T @ X / X.shape[0]


def hessian(X, theta):
    h = hypothesis(theta, X)
    value = np.sum(h.T @ h)
    hessian_matrix = value * X.T @ X
    return hessian_matrix / X.shape[0]


def gradient_reg(X, y, theta, lm):
    grad = (hypothesis(theta, X).reshape(-1, 1) - y.reshape(-1, 1)).T @ X / X.shape[0]
    return grad + (lm / X.shape[0]) * theta


def lr_grad_descent(X, y, alpha, threshold, is_reg, lm=1e-4):
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    theta = np.random.randn(X.shape[1])
    cost_last = cost_function(X, y, theta)
    cost = [cost_last]
    iter_num = 0
    while True:
        if is_reg == 0:
            grad = gradient(X, y, theta)
            theta = theta - alpha * grad
            cost_now = cost_function(X, y, theta)
            if cost_last - cost_now < threshold:
                break
            print(cost_now)
            cost.append(cost_now)
            cost_last = cost_now
            iter_num = iter_num + 1
        else:
            grad = gradient_reg(X, y, theta, lm)
            theta = theta - alpha * grad
            cost_now = cost_function_reg(X, y, theta, lm)
            if cost_last - cost_now < threshold:
                break
            print(cost_now)
            cost.append(cost_now)
            cost_last = cost_now
            iter_num = iter_num + 1
    print("共迭代" + str(iter_num) + "次")
    return theta, cost, iter_num


def lr_newton(X, y, threshold):
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    theta = np.random.randn(X.shape[1]).reshape(1, -1)
    cost_last = cost_function(X, y, theta)
    cost = [cost_last]
    iter_num = 0
    while True:
        grad = gradient(X, y, theta)
        hessian_matrix = hessian(X, theta)
        theta = theta - grad @ np.linalg.pinv(hessian_matrix)
        cost_now = cost_function(X, y, theta)
        cost.append(cost_now)
        print(cost_now)
        iter_num = iter_num + 1
        if cost_last - cost_now < threshold:
            break
        cost_last = cost_now
    print("共迭代" + str(iter_num) + "次")
    return theta, cost, iter_num


def draw(theta):
    test_data = np.linspace(-2, 2, 100)
    plt.figure(1)
    plt.plot(test_data, -(theta[:, 0] + theta[:, 1] * test_data) / theta[:, 2])
    plt.show()


def main():
    X, y = create_data_naive_bayesian(1000, 1)
    alpha = 1e-3
    threshold = 1e-8
    theta, cost, iter_num = lr_grad_descent(X, y, alpha, threshold, 0)        # 梯度下降法，带正则项
    # theta, cost, iter_num = lr_newton(X, y, threshold)                        # 牛顿法
    draw(theta)


if __name__ == '__main__':
    main()
