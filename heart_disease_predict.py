from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logistic_regression as lr
import pandas as pd
import numpy as np


def logistic_UCI_data():
    column_1 = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]
    column_2 = ["thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    column = column_1 + column_2
    data = pd.read_csv("data/processed.cleveland.csv", engine='python', names=column)
    data = (data.replace(to_replace="?", value=np.nan)).dropna()
    X_train, X_test, y_train, y_test = train_test_split(data[column[0:13]], data[column[13]], test_size=0.25)
    std = StandardScaler()

    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    theta, cost, iter_num = lr.lr_grad_descent(X_train, y_train.array.to_numpy(), 1e-5, 1e-8, 1, lm=1e-6)
    print(theta)
    y_predict = fit(theta, X_test)
    rate = cal_rate(y_predict, y_test)
    fit(theta, X_test)
    print("正确率为" + str(rate))


def fit(theta, X_test):
    X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1)
    y = lr.sigmoid(theta @ X_test.T)
    y_predict = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        if np.sum(y[:, i]) >= 0.5:
            y_predict[i] = 1
        else:
            y_predict[i] = 0
    return y_predict


def cal_rate(y_predict, y_test):
    n = y_predict.size
    count = 0
    for i in range(n):
        if y_predict[i] == y_test.array[i] and y_predict[i] == 0:
            count = count + 1
        if y_predict[i] > 0 and y_test.array[i] > 0:
            count = count + 1
    return count / n


def main():
    logistic_UCI_data()


if __name__ == '__main__':
    main()
