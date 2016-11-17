# coding=utf-8

import pandas as pd
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('./data/SMSSpamCollection', delimiter='\t', header=None)
    print(df.head())
    print('含spam短信数量：', df[df[0]=='spam'][0].count())
    print('含蛤蟆短信数量：', df[df[0]=='ham'][0].count())

    ## 划分训练集和测试集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

    ## 计算TF-IDF权重
    vectoizer = TfidfVectorizer()
    # fit transform fit_transform
    X_train = vectoizer.fit_transform(X_train_raw)
    X_test = vectoizer.transform(X_test_raw)

    ## 建立逻辑回归模型
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    ## 预测
    predictions = classifier.predict(X_test)
    for i, prediction in enumerate(predictions[-5:]):
        print('预测类型：%s,信息：%s'%(prediction, X_test_raw.iloc[i]))