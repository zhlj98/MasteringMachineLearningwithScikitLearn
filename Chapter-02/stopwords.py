# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == '__main__':
    ## 将文档转换成词向量
    corpus = ['UNC played Duke in basketball',
              'Duke lost the basketball game',
              'I ate a sandwich']
    vectorizer = CountVectorizer(stop_words='english')
    print(vectorizer.fit_transform(corpus).todense())
    print(vectorizer.vocabulary_)