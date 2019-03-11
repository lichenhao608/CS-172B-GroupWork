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

# load the input data
with open('test2.txt') as f:
    sentence = f.readlines()
x_train_raw = sentence[:901]
x_test_raw = sentence[901:]

# load the target data
target = []
with open('test3.txt') as f:
    for i in f:
        target.append(int(i))
y_train = target[:901]
y_test = target[901:]

# word embedding
tok = Tokenizer()
tok.fit_on_texts(set(sum([x.split() for x in sentence], [])))
print(tok.document_count)

# find the max sentence(we can set by ourselves if we know the number)
max_len = max([len(x.split()) for x in sentence])
vocab_size = len(tok.word_index) + 1
x_train = pad_sequences(tok.texts_to_sequences(
    x_train_raw), maxlen=max_len, padding='post')
x_test = pad_sequences(tok.texts_to_sequences(
    x_test_raw), maxlen=max_len, padding='post')


'''
conv1d(output size, filter size)
结构：
    word embedding: 因为这组dataset最多就31个单词，所以这里dimension只用10
                    在正式做CNN前可以加一层dropout，也能有效反正overfitting

    1层convolution 每层结尾 maxpooling
    Flatten  (用于maxpooling后连接到dense layer的转变)
    1层dense
    dropout  (.3 的掉率,用于防止overfitting.)
    ouput    (用softmax)
'''
embedding_dim = 10
model = Sequential()

# word embedding
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Dropout(0.1))

# layer 1
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(5))

# layer 2
# model.add(Conv1D(64, 5, activation = 'relu'))
# model.add(MaxPooling1D(5))

# layer 3
# model.add(Conv1D(64, 5, activation = 'relu'))
# model.add(MaxPooling1D(5))

# flatten
model.add(Flatten())

# dense
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# training and get score
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=10, epochs=10)
score = model.evaluate(x_train, y_train)
print("Training Accuracy: {:.4f}".format(score[1]))
score = model.evaluate(x_test, y_test)
print("Testing Accuracy: {:.4f}".format(score[1]))
