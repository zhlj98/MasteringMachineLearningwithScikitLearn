# coding=utf-8

from sklearn.feature_extraction.text import HashingVectorizer

if __name__ == '__main__':
    corpus = ['the', 'ate', 'bacon', 'cat']
    vectorizer = HashingVectorizer(n_features=6)
    print(vectorizer.transform(corpus).todense())