import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt


def load_data(data):

    data = np.loadtxt("data/" + data)
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y


if __name__ == "__main__":

    X, y = load_data('prehopper.txt')
    
    model = Sequential()
    rbflayer = RBFLayer(64,
                        initializer=InitCentersRandom(X),
                        betas=2.0,
                        input_shape=(16,)) #16,72
    model.add(rbflayer)
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam())
    
    model.fit(X, y,
              batch_size=256,
              epochs=1000,
              verbose=1)


    model.save('rbfhopper.h5')

