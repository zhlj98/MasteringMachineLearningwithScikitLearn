# coding=utf-8

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./data/ad.data', header=None, low_memory=False)
    explanatory_variable_columns = set(df.columns.values)
    response_variable_column = df[len(df.columns.values)-1]
    explanatory_variable_columns.remove(len(len(df.columns.values)-1))

    y = [1 if e=='ad.' else 0 for e in response_variable_column]
    X = df.loc[:, list(explanatory_variable_columns)]

    X.replace(to_replace='*\?', value=-1, regex=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipline = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])

    parameters = {
        'clf__max_depth':(150, 155, 160),
        'clf__min_samples_split':(1, 2, 3),
        'clf__min_samples_leaf':(1, 2, 3)
    }

    grid_search = GridSearchCV(pipline, parameters, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train, y_train)

    print('最佳效果：')
    print(grid_search.best_score_)

    print('最优参数：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s:%r'%(param_name, best_parameters[param_name]))

    predictions = grid_search.predict(X_test)
    print(classification_report(y_test, predictions))