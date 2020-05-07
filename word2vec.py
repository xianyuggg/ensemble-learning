import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import scale
from settings import EnsembleConfig

N_DIM = 256


def build_word_vec(text, size, imdb_w2v, tfidf_vocab=None, tfidf_matrix=None):
    import types
    if type(text) == type('hello'):
        from dataset import text_parse
        text = text_parse(text)
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            if tfidf_vocab is not None and tfidf_vocab is not None:
                vec += imdb_w2v[word].reshape((1, size)) * tfidf_matrix[tfidf_vocab[word]]
            else:
                vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def get_TFIDF(texts_with_tokenize):
    # corpus = [
    #     'This is the first document.',
    #     'This is the second second document.',
    #      https://www.jianshu.com/p/c7e2771eccaa
    # ]
    corpus = []
    for words in texts_with_tokenize:
        string = ''
        for word in words:
            string += word + ' '
        corpus.append(string)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    from dataset import stopwords
    tfidf_vec = TfidfVectorizer(lowercase=True, stop_words=stopwords, min_df=10)
    tfidf_matrix = tfidf_vec.fit_transform(corpus).toarray()

    return tfidf_vec.vocabulary_, tfidf_matrix


def get_train_vecs(x_train, x_validation, x_test, config: EnsembleConfig = None):
    all_text = x_train + x_validation + x_test
    if config.tf_idf:
        tfidf_vocab, tfidf_matrix = get_TFIDF(all_text)
    else:
        tfidf_vocab, tfidf_vocab = None, None
    if config.external_w2v:
        import logging
        from gensim.models import word2vec
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.Text8Corpus(u"raw_data/text8")  # 加载语料
        imdb_w2v = word2vec.Word2Vec(sentences, size=N_DIM)  # 训练skip-gram模型; 默认window=5
    else:
        from gensim.models.word2vec import Word2Vec
        imdb_w2v = Word2Vec(size=N_DIM, min_count=10)
        imdb_w2v.build_vocab(all_text)
        imdb_w2v.train(all_text, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)

    if not os.path.exists('model'):
        os.mkdir('model')
    imdb_w2v.save('model/w2v_model.pkl')

    idx_offset = 0
    if config.tf_idf:
        train_vecs = np.concatenate([build_word_vec(z, N_DIM, imdb_w2v, tfidf_vocab, tfidf_matrix[i + idx_offset]) for i, z in enumerate(x_train)])
        train_vecs = scale(train_vecs)
        np.save('dataset/x_train.npy', train_vecs)

        idx_offset = len(x_train)
        validation_vecs = np.concatenate([build_word_vec(z, N_DIM, imdb_w2v, tfidf_vocab, tfidf_matrix[i + idx_offset]) for i, z in enumerate(x_validation)])
        validation_vecs = scale(validation_vecs)
        np.save('dataset/x_validation.npy', validation_vecs)

        idx_offset = len(x_train) + len(x_validation)
        # Build test tweet vectors then scale
        test_vecs = np.concatenate([build_word_vec(z, N_DIM, imdb_w2v, tfidf_vocab, tfidf_matrix[i + idx_offset]) for i, z in enumerate(x_test)])
        test_vecs = scale(test_vecs)
        np.save('dataset/x_test.npy', test_vecs)
    else:
        train_vecs = np.concatenate(
            [build_word_vec(z, N_DIM, imdb_w2v) for i, z in
             enumerate(x_train)])
        train_vecs = scale(train_vecs)
        np.save('dataset/x_train.npy', train_vecs)


        validation_vecs = np.concatenate(
            [build_word_vec(z, N_DIM, imdb_w2v) for i, z in
             enumerate(x_validation)])
        validation_vecs = scale(validation_vecs)
        np.save('dataset/x_validation.npy', validation_vecs)

        # Build test tweet vectors then scale
        test_vecs = np.concatenate(
            [build_word_vec(z, N_DIM, imdb_w2v) for i, z in
             enumerate(x_test)])
        test_vecs = scale(test_vecs)
        np.save('dataset/x_test.npy', test_vecs)
