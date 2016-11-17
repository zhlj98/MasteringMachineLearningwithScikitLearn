# coding=utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./data/train.tsv', delimiter='\t')
    print(df.head())
    print(df.count())
    print(df.Phrase.head(10))
    print(df.Sentiment.describe())
    print(df.Sentiment.value_counts())

    pipeline = Pipeline([('vect', TfidfVectorizer(stop_words='english')),
                         ('clf', LogisticRegression())])
    parameters = {'vect__max_df':(0.25, 0.5),
                  'vect__ngram_range':((1, 1), (1, 2)),
                  'vect__use_idf':(True, False),
                  'clf__C':(0.1, 1, 10)}

    X, y = df['Phrase'], df['Sentiment'].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print('最佳效果：%0.3f'%grid_search.best_score_)
    print('最优参数组合：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s:%r'%(param_name, best_parameters[param_name]))

    ## 模型评估
    predicions = grid_search.predict(X_test)
    print('准确率：', accuracy_score(y_test, predicions))
    print('混淆矩阵：', confusion_matrix(y_test, predicions))
    print('分类报告：', classification_report(y_test, predicions))