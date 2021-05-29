import numpy as np
from scipy.special import expit

class Sigmoid:

    @staticmethod
    def activation(x):
        return expit(x)
    @staticmethod
    def activation_derivated(x):
        return x * (1-x)

class Relu:

    @staticmethod
    def activation(x):
        return np.maximum(x, 0) 
    @staticmethod
    def activation_derivated(x):
        return np.array(x >= 0).astype('int')