# coding-utf-8

from sklearn.feature_extraction import DictVectorizer

if __name__ == '__main__':
    onehot_encoder = DictVectorizer()
    instances = [{'city':'New York'},
                 {'city':'San Francisco'},
                 {'city':'Chapel Hill'}]
    print(onehot_encoder.fit_transform(instances).toarray())