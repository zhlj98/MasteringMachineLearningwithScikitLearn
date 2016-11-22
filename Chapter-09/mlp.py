# coding=utf-8

from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MultilayerPerceptronClassifier
import numpy as np

if __name__ == '__main__':

    y = [0, 1, 1, 0] * 1000
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
    clf = MultilayerPerceptronClassifier(hidden_layer_sizes=[2],
                                         activation='logistic',
                                         algorithm='sgd',
                                         random_state=3)
    clf.fit(X_train, y_train)

    print('层数：%s, 输出单元数量：%s'%(clf.n_layers_, clf.n_outputs_))
    predictions = clf.predict(X_test)
    print('准确率：%s'%clf.score(X_test, y_test))
    for i, p in enumerate(predictions[:10]):
        print('真实值：%s, 预测值：%s'%(y_test[i], p))