# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
    vectorizer = CountVectorizer(stop_words='english')
    print(vectorizer.fit_transform(corpus).todense())
    print(vectorizer.vocabulary_)

    corpus = ['The dog ate a sandwich and I ate a sandwich',
              'The wizard transfigured a sandwich']
    vectorizer = TfidfVectorizer(stop_words='english')
    print(vectorizer.fit_transform(corpus).todense())
    print(vectorizer.vocabulary_)