import numpy as np


class neural_layer():
  """ parametros de entrada:
      n_conn = numero de conexiones con la capa anterior
      n_neur = numero de neuronas que tendra esta capa
      act_f = funcion de activacion que tendra esta capa
  """ 
  def __init__(self, n_conn, n_neur, act_f):
    self.act_f = act_f
    """ vector columna con numero de bias, el cual coincide con el numero de neuronas
        el cual partira con valores al azar de -1 a 1
    """
    self.bias = np.random.rand(1, n_neur) * 2 - 1
    """ matriz con los pesos asociados entre la capa anterior y esta
    """
    self.weights = np.random.rand(n_conn, n_neur) * 2 - 1