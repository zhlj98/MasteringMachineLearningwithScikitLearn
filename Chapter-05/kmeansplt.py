# coding=utf-8

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X0 = np.array([7, 5, 7, 3, 4, 1, 0, 2, 8, 6, 5, 3])
    X1 = np.array([5, 7, 7, 3, 6, 4, 0, 2, 7, 8, 5, 7])
    plt.figure()
    plt.axis([-1, 9, -1, 9])
    plt.grid(True)
    plt.plot(X0, X1, 'k.')
    plt.show()

    C1 = [1, 4, 5, 9, 11]
    C2 = list(set(range(12))-set(C1))
    X0C1, X1C1 = X0[C1], X1[C1]
    X0C2, X1C2 = X0[C2], X1[C2]
    plt.figure()
    plt.title('one')
    plt.axis([-1, 9, -1, 9])
    plt.grid(True)
    plt.plot(X0C1, X1C1, 'rx')
    plt.plot(X0C2, X1C2, 'g.')
    plt.plot(4, 6, 'rx', ms=12.0)
    plt.plot(5, 5, 'g.', ms=12.0)
    plt.show()

    C1 = [1, 2, 4, 8, 9, 11]
    C2 = list(set(range(12)) - set(C1))
    X0C1, X1C1 = X0[C1], X1[C1]
    X0C2, X1C2 = X0[C2], X1[C2]
    plt.figure()
    plt.title('two')
    plt.axis([-1, 9, -1, 9])
    plt.grid(True)
    plt.plot(X0C1, X1C1, 'rx')
    plt.plot(X0C2, X1C2, 'g.')
    plt.plot(3.8, 6.4, 'rx', ms=12.0)
    plt.plot(4.57, 4.14, 'g.', ms=12.0)
    plt.show()

    C1 = [0, 1, 2, 4, 8, 9, 10, 11]
    C2 = list(set(range(12)) - set(C1))
    X0C1, X1C1 = X0[C1], X1[C1]
    X0C2, X1C2 = X0[C2], X1[C2]
    plt.figure()
    plt.title('three')
    plt.axis([-1, 9, -1, 9])
    plt.grid(True)
    plt.plot(X0C1, X1C1, 'rx')
    plt.plot(X0C2, X1C2, 'g.')
    plt.plot(5.5, 7.0, 'rx', ms=12.0)
    plt.plot(2.2, 2.8, 'g.', ms=12.0)
    plt.show()

    ## 肘部法则
    cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
    cluster2 = np.random.uniform(3.5, 4.5, (2, 10))

    X = np.hstack((cluster1, cluster2)).T
    plt.figure()
    plt.axis([0, 5, 0, 5])
    plt.grid(True)
    plt.plot(X[:, 0], X[:, 1], 'k.')
    plt.show()

    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('ji bian')
    plt.title('best k')
    plt.show()