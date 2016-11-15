# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, cross_val_score

if __name__ == '__main__':
    df = pd.read_csv('data/winequality-red.csv', sep=';')
    print(df.head())
    print(df.describe())

    plt.scatter(df['alcohol'], df['quality'])
    plt.xlabel('Alcohol')
    plt.ylabel('Quality')
    plt.title('Alcohol and Quality')
    plt.show()

    X = df[list(df.columns)[:-1]]
    y = df['quality']

    ## 使用交叉验证分离样本
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_predictions = regressor.predict(X_test)
    print('R-squared:', regressor.score(X_test, y_test))

    ## 使用交叉验证进行多验证
    scores = cross_val_score(regressor, X, y, cv=5)
    print(scores.mean(), scores)

    ## 模型的预测品质与实际品质的图像
    plt.scatter(y_test, y_predictions)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('Actual and Predicted')
    plt.show()