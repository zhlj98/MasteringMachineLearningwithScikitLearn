# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    X = np.array([[0.2, 0.1],
                  [0.4, 0.6],
                  [0.5, 0.2],
                  [0.7, 0.9]])
    y = [0, 0, 0, 1]

    markers=['.', 'x']
    plt.scatter(X[:3, 0], X[:3, 1], marker='.', s=400)
    plt.scatter(X[3, 0], X[3, 1], marker='x', s=400)
    plt.xlabel('sleep')
    plt.ylabel('angry')
    plt.show()