# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == '__main__':
    ## 将文档转换成词向量
    corpus = ['UNC played Duke in basketball',
              'Duke lost the basketball game',
              'I ate a sandwich']
    vectorizer = CountVectorizer()
    print(vectorizer.fit_transform(corpus).todense())
    print(vectorizer.vocabulary_)

    ## 计算任意两个文档之间的相似度
    counts = vectorizer.fit_transform(corpus).todense()
    for x, y in [[0, 1], [0, 2], [1, 2]]:
        dist = euclidean_distances(counts[x], counts[y])
        print('文档{}与文档{}的距离{}'.format(x, y, dist))