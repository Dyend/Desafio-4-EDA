import layers as ls
import neuronal_network
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from activation import Sigmoid
from loss import MSE
from sklearn.datasets import make_circles

def test():
    # la intencion de este programa es replicar lo de dotcsv con variables mas descriptivas para entenderlo mejor.
    # crear dataset
    numero_muestras = 500
    caracteristicas = 2
    # factor es la distancia entre los 2 circulos
    entrada, salida = make_circles(n_samples=numero_muestras, factor=0.5, noise=0.09)

    salida = salida[:, np.newaxis]
    ''' print(entrada[salida == 0, 0], entrada[salida == 0, 1])
    print('---------------------------------------------------')
    print(entrada[salida == 1, 0], entrada[salida == 1, 1]) '''
    plt.scatter(entrada[salida[:, 0] == 0, 0], entrada[salida[:, 0] == 0, 1])
    plt.scatter(entrada[salida[:, 0] == 1, 0], entrada[salida[:, 0] == 1, 1])
    plt.axis("equal")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    topology = [caracteristicas, 8, 4, 1]
    nn = neuronal_network.Network(topology, Sigmoid, MSE)
    learning_rate = 0.1
    nn.train(entrada, salida, learning_rate)

if __name__ == '__main__':
    test()
    """ se lee el dataset y se pre procesa"""
    #data = read_data(path)

    """ Topology es una lista con la cantida de neuronas en cada capa
        act_f es la funcion de activacion que sera usada por cada capa """
    test()
    """ topology = [8, 8, 4, 1]
    act_f = activation.Sigmoid

    print('Creando red neuronal de topologia: ', topology)
    entry_layer = ls.EntryLayer(topology[0])
    layers = []
    layers.append(entry_layer)
    for index, output_size in enumerate(topology[1:]):
        input_size = topology[index - 1]
        layers.append(ls.FullyConnected(input_size, output_size, act_f))

    network = neuronal_network.Network(layers, MSE) """
