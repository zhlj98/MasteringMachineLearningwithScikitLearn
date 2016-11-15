# coding=utf-8
from sklearn.linear_model import LinearRegression
import numpy as np

if __name__ == '__main__':
    ## 通过Numpy矩阵操作完成
    X = [[1, 6, 2],
         [1, 8, 1],
         [1, 10, 0],
         [1, 14, 2],
         [1, 18, 0]]
    y = [[7], [9], [13], [17.5], [18]]
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
    print(w)

    ## Numpy矩阵也提供了最小二乘法来实现
    w = np.linalg.lstsq(X, y)[0]
    print(w)

    ## 使用sklearn
    X = [[6, 2],
         [8, 1],
         [10, 0],
         [14, 2],
         [18, 0]]
    y = [[7], [9], [13], [17.5], [18]]
    model = LinearRegression()
    model.fit(X, y)

    X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
    y_test = [[11], [9], [13], [17.5], [18]]

    predictions = model.predict(X_test)
    for i, prediction in enumerate(predictions):
        print('Predicted: %s, Target: %s'%(prediction, y_test[i]))

    print('R-squared:%.2f'%model.score(X_test, y_test))