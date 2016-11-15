# coding=utf-8

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    data = load_boston()

    ## 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

    ## 做归一化处理
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    X_test = X_scaler.fit_transform(X_test)
    y_test = y_scaler.fit_transform(y_test)

    ## 用交叉方法完成训练和测试
    regressor = SGDRegressor(loss='squared_loss')
    scores = cross_val_score(regressor, X_train, y_train, cv=5)
    print('交叉验证R方值：', scores)
    print('交叉验证R方均值：', np.mean(scores))
    regressor.fit_transform(X_train, y_train)
    print('测试集R方值：', regressor.score(X_test, y_test))
