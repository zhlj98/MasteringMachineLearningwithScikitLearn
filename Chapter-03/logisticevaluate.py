# coding=utf-8

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    ## 混淆矩阵
    y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    plt.matshow(confusion_matrix)
    plt.colorbar()
    plt.show()

    ## 准确率
    print(accuracy_score(y_test, y_pred))

    df = pd.read_csv('./data/sms.csv')
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print('准确率：', np.mean(scores), scores)

    ## 精确率
    precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
    print('精确率：', np.mean(precisions), precisions)

    ## 召回率
    recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
    print('召回率：', np.mean(recalls), recalls)

    ## F1度量
    f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
    print('综合评价：', np.mean(f1s), f1s)

    ## ROC AUC
    precisions = classifier.predict_proba(X_test)
    false_positive_rate, recall, thresholds = roc_curve(y_test, precisions[:,1])
    roc_auc = auc(false_positive_rate, recall)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f'%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()