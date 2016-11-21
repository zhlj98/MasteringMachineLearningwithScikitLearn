# coding=utf-8

from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':

    digits = fetch_mldata('MNIST original', data_home='./data/mnist').data

    counter = 1
    for i in range(1, 4):
        for j in range(1, 6):
            plt.subplot(3, 5, counter)
            plt.imshow(digits[(i-1)*8000+j].reshape((28, 28)), cmap=cm.Greys_r)
            plt.axis('off')
            counter += 1
    plt.show()

    X, y = digits.data, digits.target
    X = X/255.0*2-1
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline([('clf', SVC(kernel='rbf', gamma=.001, C=100))])
    parameters = {'clf_gamma': (0.01, 0.03, 0.1, 0.3, 1),
                  'clf_C': (0.1, 0.3, 1, 3, 10, 30)}
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', refit=True)
    grid_search.fit(X_train, y_train)

    print('最佳效果：%0.3f'%grid_search.best_score_)
    print('最优参数集：')

    best_prameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s:%r'%(param_name, best_prameters[param_name]))
    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))