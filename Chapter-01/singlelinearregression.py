# coding=utf-8

from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def runplt():
    plt.figure()
    # plt.title('披萨价格与直径数据')
    # plt.xlabel('直径（英寸）')
    # plt.ylabel('价格（美元）')
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt

if __name__ == '__main__':
    ## 用matplotlib绘制图形
    #font = FontProperties(fname=r'C:\windows\fonts\msyh.ttc', size=10)
    plt = runplt()
    X = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    plt.plot(X, y, 'k.')
    plt.show()

    ## 用scikit-learn构建模型
    # 创建并拟合模型
    model = LinearRegression()
    model.fit(X, y)
    print('预测一张12英寸披萨价格：$%.2f' % model.predict([12])[0])

    ## 绘制拟合直线
    X2 = [[0], [10], [14], [25]]
    y2 = model.predict(X2)
    plt.plot(X, y, 'k.')
    plt.plot(X2, y2, 'g-')
    plt.show()

    ## 绘制残差
    X2 = [[0], [10], [14], [25]]
    y2 = model.predict(X2)
    plt.plot(X, y, 'k.')
    plt.plot(X2, y2, 'g-')

    yr = model.predict(X)
    for idx, x in enumerate(X):
        plt.plot([x, x], [y[idx], yr[idx]], 'r-')
    plt.show()

    ## 使用R方评估预测效果
    X_test = [[8], [9], [11], [16], [12]]
    y_test = [[11], [8.5], [15], [18], [11]]
    r = model.score(X_test, y_test)
    print(r)