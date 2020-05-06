from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('model/w2v_model.pkl')

print(Word2Vec.most_similar(model, 'happy', topn=5))

