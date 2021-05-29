import pandas as pd
import matplotlib
import numpy as np
import os
import pickle
from numpy.core.fromnumeric import shape
from matplotlib import pyplot as plt

def read_data(path, unwanted_cols):
    cols = list(pd.read_csv(path, nrows =0))
    # [1:] para quitar la columna con el indice
    cols = [i for i in cols if not i in unwanted_cols][1:]
    data = pd.read_csv(path, sep=',', header=0, usecols=cols).values
    return data, cols

# guarda el grafico de loss en un png 
def plot_loss(loss, path):
    matplotlib.use('Agg')
    plt.plot(range(len(loss)), loss)
    if not os.path.isfile(path):
        os.makedirs('out')
    plt.savefig(path)

""" Separa en entrada y salida para la red neuronal, salida columns identifica cuales columnas son la salida
    el parametro salida columns es una lista con los indices correspondientes a las columnas de salida """
def split_data(data, salida_columns):

    _Y = np.zeros(shape=(len(data[0]), 1))
    _Y = data[:, salida_columns]
    _X = np.delete(data, salida_columns, axis=1)
    return _X, _Y

# retorna una tupla con la data para trainear y para testear
def pre_proc(data, training_percent):
    # Shuffle data
    for i in range(len(data)):
        np.random.shuffle(data)
    # Normalize Data
    # If data contains negative values we would need to subtract the minimum first.
    data = (data-data.min(axis=0))/ data.ptp(0)
    # Then we normalize it.
    data = data / data.max(axis=0)
    # Finally we divide the data we will use for training the neuronal network and the data we will use for testing it.
    training, test = data[:training_percent,:], data[training_percent:,:]

    return training, test

def save_nn(nn, path):
    with open(path, 'wb') as output:
        pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)

def load_nn(path):
    with open(path, 'rb') as input:
        return pickle.load(input)