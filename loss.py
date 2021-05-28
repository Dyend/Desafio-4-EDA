
import numpy as np

class MSE:
    '''
        Clase que define la funcion de perdida
        para el error cuadratico medio junto con su derivada
    '''
    @staticmethod
    def loss(real, prediction):
        return np.mean((prediction - real) ** 2)
    @staticmethod
    def derivated_loss(prediction, real):
        return (prediction - real)