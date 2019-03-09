import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential,Model #new
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD

#load the input data
with open('test2.txt') as f:
    sentence = f.readlines()
x_train_raw = sentence[:901]
x_test_raw = sentence[901:]

#Read data
target = []
with open('test3.txt') as f:
    for i in f:
        target.append(int(i))
y_train = target[:901]
y_test = target[901:]

corpus = [x for x in sentence]
tok = Tokenizer()
tok.fit_on_texts(corpus)


max_len = max([len(x.split()) for x in sentence])
vocab_size = len(tok.word_index) + 1
x_train = pad_sequences(tok.texts_to_sequences(x_train_raw), maxlen = max_len, padding = 'post')
x_test = pad_sequences(tok.texts_to_sequences(x_test_raw), maxlen = max_len, padding = 'post')

embedding_dim = 10
model = Sequential()

#STRUCTURE:
'''
embedding
LSTM
Dropout
Dense with softmax
'''
model.add(Embedding(vocab_size, embedding_dim, input_length = max_len))
model.add(LSTM(units=300, activation = 'tanh', kernel_initializer = 'glorot_uniform'))
model.add(Dropout(0.1))
model.add(Dense(vocab_size,activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 128, epochs = 10,verbose=1)
score = model.evaluate(x_train, y_train)
print("Training Accuracy: {:.4f}".format(score[1])) #Training Accuracy: 0.5083
score = model.evaluate(x_test, y_test)
print("Testing Accuracy: {:.4f}".format(score[1]))  #Testing Accuracy: 0.4242
        
