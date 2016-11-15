# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def runplt():
    plt.figure()
    # plt.title('披萨价格与直径数据')
    # plt.xlabel('直径（英寸）')
    # plt.ylabel('价格（美元）')
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt

if __name__ == '__main__':
    X_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]
    X_test = [[6], [8], [11], [16]]
    y_test = [[8], [12], [15], [18]]

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    xx =np.linspace(0, 26, 100)
    yy = regressor.predict(xx.reshape(xx.shape[0], 1))
    plt = runplt()
    plt.plot(X_train, y_train, 'k.')
    plt.plot(xx, yy)

    pf = PolynomialFeatures(degree=2)
    X_train_pf = pf.fit_transform(X_train)
    X_test_pf = pf.transform(X_test)
    xx_pf = pf.transform(xx.reshape(xx.shape[0], 1))
    regressor_pf = LinearRegression()
    regressor_pf.fit(X_train_pf, y_train)
    plt.plot(xx, regressor_pf.predict(xx_pf), 'r-')

    pf_cubic = PolynomialFeatures(degree=3)
    X_train_cubic = pf_cubic.fit_transform(X_train)
    X_test_cubic = pf_cubic.transform(X_test)
    regressor_cubic = LinearRegression()
    regressor_cubic.fit(X_train_cubic, y_train)
    xx_cubic = pf_cubic.transform(xx.reshape(xx.shape[0], 1))
    plt.plot(xx, regressor_cubic.predict(xx_cubic))
    plt.show()

    print(X_train)
    print(X_train_pf)
    print(X_test)
    print(X_test_pf)
    print(X_train_cubic)
    print(X_test_cubic)
    print('一元线性回归r-squared', regressor.score(X_test, y_test))
    print('二次回归r-squared', regressor_pf.score(X_test_pf, y_test))
    print('三次回归r-squared', regressor_cubic.score(X_test_cubic, y_test))