import keras
import tensorflow as tf
import numpy as np
import random
from collections import Counter
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Embedding, LSTM
from keras.layers import Bidirectional, BatchNormalization, SpatialDropout1D
from nltk.tokenize.regexp import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD

text, y = [], []
with open('sentiment labelled sentences\\sentiment labelled sentences\\amazon_cells_labelled.txt') as f:
    a = f.readlines()

random.shuffle(a)
for t in a:
    c, b = t.split('\t')
    text.append(c)
    y.append(int(b))

tok = RegexpTokenizer(r'\w+|\!|\?|\.')

vocab = Counter()


def text_to_wordlist(text, lower=True):
    '''
    convert single sentence into a list of words that contains punctuation !?.
    '''
    text = tok.tokenize(text)

    if lower:
        text = [i.lower() for i in text]

    vocab.update(text)
    return text


def process_sentence(sentences, lower=True):
    return [text_to_wordlist(sent, lower=lower) for sent in sentences]


comment = process_sentence(text)

print(len(vocab))

# Word2Vec vetorizing words
model = Word2Vec(comment, workers=16)
wordvec = model.wv
print(len(wordvec.vocab))

max_nb_word = len(wordvec.vocab)
max_len = max(len(i) for i in comment)

word_index = {t[0]: i + 1 for i,
              t in enumerate(vocab.most_common(max_nb_word))}

sequences = [[word_index.get(t, 0) for t in c]
             for c in comment[:900]]
test_sequences = [[word_index.get(t, 0) for t in c]
                  for c in comment[900:]]

data = pad_sequences(sequences, maxlen=max_len, padding='post')
test_data = pad_sequences(test_sequences, maxlen=max_len, padding='post')
y_train = y[:900]
y_test = y[900:]

wv_dim = 100
nb_words = min(max_nb_word, len(wordvec.vocab))+1
# we initialize the matrix with random numbers
wv_matrix = (np.random.rand(nb_words, wv_dim) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_nb_word:
        continue
    try:
        embedding_vector = wordvec[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass

# Bidirectional Neural Networks
nn_model = Sequential()
nn_model.add(Embedding(nb_words, wv_dim, mask_zero=False, weights=[
             wv_matrix], input_length=max_len))
nn_model.add(SpatialDropout1D(0.2))
nn_model.add(Bidirectional(LSTM(15)))
nn_model.add(Dropout(0.1))

# nn_model.add(BatchNormalization())
nn_model.add(Dense(2, activation='softmax'))
nn_model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam', metrics=['accuracy'])

history = nn_model.fit(data, y_train, batch_size=10, epochs=10)
score = nn_model.evaluate(data, y_train)
print("Training Accuracy: {:.4f}".format(score[1]))
score = nn_model.evaluate(test_data, y_test)
print("Testing Accuracy: {:.4f}".format(score[1]))
