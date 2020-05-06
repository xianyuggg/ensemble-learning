import pandas as pd
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
import os
from sklearn.preprocessing import scale

N_DIM = 128


def build_word_vec(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def get_train_vecs(x_train, x_validation, x_test):
    all_text = x_train + x_validation + x_test
    imdb_w2v = Word2Vec(size=N_DIM, min_count=10)
    imdb_w2v.build_vocab(all_text)

    imdb_w2v.train(all_text, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)

    if not os.path.exists('model'):
        os.mkdir('model')
    imdb_w2v.save('model/w2v_model.pkl')

    train_vecs = np.concatenate([build_word_vec(z, N_DIM, imdb_w2v) for z in x_train])
    train_vecs = scale(train_vecs)
    np.save('dataset/x_train.npy', train_vecs)

    validation_vecs = np.concatenate([build_word_vec(z, N_DIM, imdb_w2v) for z in x_validation])
    validation_vecs = scale(validation_vecs)
    np.save('dataset/x_validation.npy', validation_vecs)

    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_word_vec(z, N_DIM, imdb_w2v) for z in x_test])
    test_vecs = scale(test_vecs)
    np.save('dataset/x_test.npy', test_vecs)
