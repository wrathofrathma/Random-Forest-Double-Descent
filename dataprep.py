import numpy as np
from tensorflow.keras.datasets import mnist
import random


def import_mnist():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    return (x_train, y_train), (x_test, y_test)


def add_noise(labels, opts, noise):
    n_labels = len(labels) - 1
    n_opts = len(opts) - 1
    new_labels = []
    n_noise = 0.0
    for y in labels:
        if random.randint(0, 100) < noise:
            n_noise += 1.0
            ind = opts.index(y)
            new_index = random.randint(0, n_opts)
            while new_index == ind:
                new_index = random.randint(0, n_opts)
            new_labels += [opts[new_index]]
        else:
            new_labels += [y]
    return (new_labels, n_noise / n_labels)


def only4_9(data, labels):
    n_labels = []
    n_data = []
    for x, y in zip(data, labels):
        if y == 4 or y == 9:
            n_data += [x]
            n_labels += [y]
    return (n_data, n_labels)


def prep_mnist():
    (xtrain, ytrain), (xtest, ytest) = import_mnist()
    xtrain = xtrain.reshape(60000, 784)
    xtest = xtest.reshape(10000, 784)
    (xtrain, ytrain) = only4_9(xtrain, ytrain)
    (xtest, ytest) = only4_9(xtest, ytest)
    return (xtrain, ytrain), (xtest, ytest)
