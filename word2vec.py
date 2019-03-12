import os
from collections import Counter
import random
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Embedding, CuDNNLSTM
from keras.layers import Bidirectional, BatchNormalization, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from nltk.tokenize.regexp import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.optimizers import SGD

gpuid = 0  # An index of which gpu to use.
os.environ['KERAS_BACKEND'] = 'tensorflow'
# (Empty) List of gpu indices that TF can see.
os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpuid)
# Only use a single GPU.
CONFIG = tf.ConfigProto(
    device_count={'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
# Prevents tf from grabbing all gpu memory.
CONFIG.gpu_options.allow_growth = True
sess = tf.Session(config=CONFIG)
K.set_session(sess)


text, y = [], []
with open('sentiment labelled sentences\\sentiment labelled sentences\\amazon_cells_labelled.txt') as f:
    a = f.readlines()

# random.shuffle(a)
for t in a:
    c, b = t.split('\t')
    text.append(c)
    y.append(int(b))

# Simple Word embedding method
# word embedding
# tok = Tokenizer()
# tok.fit_on_texts(set(sum([x.split() for x in text], [])))
# print(tok.document_count)

# # find the max sentence(we can set by ourselves if we know the number)
# max_len = max([len(x.split()) for x in text])
# nb_words = len(tok.word_index) + 1
# data = pad_sequences(tok.texts_to_sequences(
#     text), maxlen=max_len, padding='post')

# Word2Vec Embedding
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


'''
sequences = [[word_index.get(t, 0) for t in c]
             for c in comment[:900]]
test_sequences = [[word_index.get(t, 0) for t in c]
                  for c in comment[900:]]
'''
sequences = [[word_index.get(t, 0) for t in c]
             for c in comment]
data = pad_sequences(sequences, maxlen=max_len, padding='post')

x_train = data[:900]
x_test = data[900:]
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

print(max_len)
# Bidirectional Neural Networks
nn_model = Sequential()
nn_model.add(Embedding(nb_words, wv_dim, mask_zero=False, weights=[
    wv_matrix], input_length=max_len))
# nn_model.add(Embedding(nb_words, wv_dim,
#                        mask_zero=False, input_length=max_len))

nn_model.add(SpatialDropout1D(0.2))
nn_model.add((CuDNNLSTM(15)))
nn_model.add(Dropout(0.1))

# nn_model.add(BatchNormalization())
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy',
                 optimizer='adam', metrics=['accuracy'])
'''
# Convelutional Neural Networks
nn_model.add(Dropout(0.1))

# layer 1
nn_model.add(Conv1D(64, 5, activation='relu'))
nn_model.add(MaxPooling1D(5))

# layer 2
# model.add(Conv1D(64, 5, activation = 'relu'))
# model.add(MaxPooling1D(5))

# layer 3
# model.add(Conv1D(64, 5, activation = 'relu'))
# model.add(MaxPooling1D(5))

# flatten
nn_model.add(Flatten())

# dense
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dropout(0.3))
nn_model.add(Dense(1, activation='sigmoid'))

# training and get score
nn_model.compile(loss='binary_crossentropy',
                 optimizer='adam', metrics=['accuracy'])
'''
print(nn_model.summary())
for layer in nn_model.layers:
    print(layer.input_shape)
history = nn_model.fit(data, y, validation_split=0.2,
                       batch_size=30, epochs=10)
score = nn_model.evaluate(x_train, y_train)
print("Training Accuracy: {:.4f}".format(score[1]))
score = nn_model.evaluate(x_test, y_test)
print("Testing Accuracy: {:.4f}".format(score[1]))
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
