import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD

from gensim.models import Word2Vec

import prepro

text, y, vocab = prepro.prepross(
    'sentiment labelled sentences\\sentiment labelled sentences\\amazon_cells_labelled.txt')


x_train = text[:901]
x_test = text[901:]

y_train = y[:901]
y_test = y[901:]

'''
tok = Tokenizer(filters='',lower=False)
tok.fit_on_texts(text)
print(tok.document_count)
'''

# Word2Vec method
w2v = Word2Vec(text, size=100, window=5, min_count=5,
               workers=16, sg=0, negative=5)
word_vector = w2v.wv
max_nb_words = len(word_vector.vocab)

# find the max sentence(we can set by ourselves if we know the number)
max_len = max([len(x) for x in text])
#vocab_size = len(tok.word_index) + 1
#x_train = pad_sequences(tok.texts_to_sequences(x_train), maxlen = max_len, padding = 'post')
x_test = pad_sequences(tok.texts_to_sequences(x_test),
                       maxlen=max_len, padding='post')

word_index = {t[0]: i+1 for i, t in enumerate(vocab.most_common(max_nb_words))}
sequences = [[word_index.get(t, 0) for t in comment] for comment in x_train]
test_sequences = [[word_index.get(t, 0) for t in comment]
                  for comment in x_test]
x_train = pad_sequences(sequences, maxlen=max_len, padding="post")
x_test = pad_sequences(test_sequences, maxlen=max_len, padding="post")

embedding_dim = 10

vocab_size = min(max_nb_words, len(word_vector.vocab))
# we initialize the matrix with random numbers
wv_matrix = (np.random.rand(vocab_size, embedding_dim) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_nb_words:
        continue
    try:
        embedding_vector = word_vector[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass

'''
model = Sequential()

#word embedding
model.add(Embedding(vocab_size, embedding_dim, weights=[wv_matrix] ,input_length = max_len))
model.add(Dropout(0.1))

#layer 1
model.add(Conv1D(64, 5, activation = 'relu'))
model.add(MaxPooling1D(5))

#layer 2
# model.add(Conv1D(64, 5, activation = 'relu'))
# model.add(MaxPooling1D(5))

#layer 3
# model.add(Conv1D(64, 5, activation = 'relu'))
# model.add(MaxPooling1D(5))

#flatten
model.add(Flatten())

#dense
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation = 'sigmoid'))
'''

wv_layer = Embedding(vocab_size, embedding_dim, mask_zero=False, weights=[wv_matrix],
                     input_length=max_len, trainable=False)

comment_input = Input(shape=(max_len,), dtype='int32')
embedded_sequence = wv_layer(comment_input)

embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)

# Output
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
preds = Dense(6, activation='sigmoid')(x)

# build the model
model = Model(inputs=[comment_input], outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
              metrics=[])

# training and get score
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=10, epochs=10)
score = model.evaluate(x_train, y_train)
print("Training Accuracy: {:.4f}".format(score[1]))
score = model.evaluate(x_test, y_test)
print("Testing Accuracy: {:.4f}".format(score[1]))
