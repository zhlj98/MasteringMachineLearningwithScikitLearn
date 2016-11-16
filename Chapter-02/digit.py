# coding=utf-8

from sklearn import datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
    digits = datasets.load_digits()
    print('Digit:', digits.target[0])
    print(digits.images[0])
    plt.figure()
    plt.axis('off')
    plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

    ## 将8*8的矩阵转换成64维向量来创建一个特征向量
    print('Feature vector:\n', digits.images[0].reshape(-1, 64))