import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from word2vec import get_train_vecs

data_file = 'raw_data/train.csv'
test_file = 'raw_data/test.csv'

stopwords = set()
with open('raw_data/stopwords.txt', 'r') as file:
    for line in file:
        stopwords.add(line.strip())
    file.close()


def text_parse(bigString):
    def is_stop_word(item):
        return item in stopwords
    import re
    try:
        list_of_tokens = re.split(r'\W', bigString)
        res = []
        for tok in list_of_tokens:
            if len(tok) < 3:
                continue
            if not is_stop_word(tok):
                res.append(tok.lower())
        return res
    except:
        return []


def divide_set():
    data_df = pd.read_csv(data_file, sep='\t')

    x = []
    y = []
    for review in data_df['reviewText']:
        x.append(text_parse(review))
    for score in data_df['overall']:
        y.append(int(score))

    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.1)

    x_test = []
    test_df = pd.read_csv(test_file, sep='\t')
    for review in test_df['reviewText']:
        x_test.append(text_parse(review))

    get_train_vecs(x_train, x_validation, x_test)   # generate word2vec and related dataset
    np.save('dataset/y_train.npy', y_train)
    np.save('dataset/y_validation.npy', y_validation)


def load_data():
    x_train = np.load('dataset/x_train.npy')
    y_train = np.load('dataset/y_train.npy')
    x_validation = np.load('dataset/x_validation.npy')
    y_validation = np.load('dataset/y_validation.npy')
    x_test = np.load('dataset/x_test.npy')
    return x_train, y_train, x_validation, y_validation, x_test


def load_test_data():
    """
    :return: x_test
    """
    return np.load('dataset/x_test.npy')
