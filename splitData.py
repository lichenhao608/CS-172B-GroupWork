import numpy as np
from numpy import asarray as arr
from numpy import atleast_2d as twod
'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.utils import np_utils


from keras.regularizers import l2
'''


def splitData(X, Y=None, train_fraction=0.80):  # split X, Y according to train fraction
    nx, _ = twod(X).shape
    ne = int(round(train_fraction * nx))

    Xtr, Xte = X[:ne], X[ne:]
    to_return = (Xtr, Xte)

    if Y is not None:
        Y = arr(Y).flatten()
        ny = len(Y)
        if ny > 0:
            assert ny == nx, 'splitData: X and Y must have the same length'
            Ytr, Yte = Y[:ne], Y[ne:]
            to_return += (Ytr, Yte)

    return to_return


if __name__ == "__main__":
    amazon = np.genfromtxt("amazon_cells_labelled.txt",
                           delimiter="\t", dtype=None, encoding='utf-8')
    # amazon = [(string,0),(string,1)....]
    tupleX, tupleY = zip(*amazon)
    #tupleX = (string1, string2, ...)
    #tupleY = (1,0,1,1,0,...)
    X = np.array(tupleX)
    Y = np.array(tupleY)
    X = twod(X).T
    Xtr, Xte, Ytr, Yte = splitData(X, Y, .75)
    """
    model = Sequential()
    model.add(Dense(256,activation='relu',input_shape=(999,),W_regularizer=12(0.001)))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    """
