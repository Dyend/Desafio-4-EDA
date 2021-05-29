import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from layers import neural_layer

class Network():

    def __init__(self, topology, act_f, loss):
        self.loss = loss.loss
        self.derivated_lost = loss.derivated_loss
        nn = []
        for l, layer in enumerate(topology[:-1]):
            nn.append(neural_layer(topology[l], topology[l + 1 ], act_f))
        self.layers = nn



    def backpropagation(self, out, salida, learning_rate):

        deltas = []
        for index in reversed(range(0, len(self.layers))):
            """ accedemos a index + 1 dado que la red neuronal no contiene la capa de entrada a diferencia de out,
                por lo que su indice esta desfasado en 1
            """
            suma_ponderada = out[index + 1][0]
            activacion = out[index + 1][1]
            # ultima capa
            if index == len(self.layers) - 1:
                derivada_parcial_activacion = self.derivated_lost(activacion, salida)
                derivada_parcial_suma_ponderada = self.layers[index].act_f.activation_derivated(activacion)
                deltas.insert(0, derivada_parcial_activacion * derivada_parcial_suma_ponderada)
            else:
                derivada_parcial_anterior = deltas[0] @ _w.T
                deltas.insert(0, derivada_parcial_anterior * self.layers[index].act_f.activation_derivated(activacion))

            _w = self.layers[index].weights

            # gradient descent
            self.layers[index].bias = self.layers[index].bias - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate
            self.layers[index].weights = self.layers[index].weights - out[index][1].T @ deltas[0] * learning_rate
        return deltas
    
    def forward(self, entrada):
        '''
            recorre las capas
            realizando el producto punto entre el vector
            de entrada y los pesos de la capa. Luego le
            agrega el sesgo para evitar linealidad
        '''
        
        out = [(None, entrada)]
        # forward pass
        for index, layer in enumerate(self.layers):
            """ suma ponderada out[-1] saca la ultima tupla ingresada es decir el resultado de la capa anterior
                out[-1][1] es la activacion
            """
            # suma_ponderada = out[-1][1] @ neural_net[index].weights + neural_net[index].bias
            # The @ operator can be used as a shorthand for np.matmul Matrix product of two arrays.
            suma_ponderada = out[-1][1] @ self.layers[index].weights + self.layers[index].bias
            # activacion
            activacion = self.layers[0].act_f.activation(suma_ponderada)
            # se guarda como tupla los valores de la suma ponderada y la activacion
            out.append((suma_ponderada, activacion))
        return out

    def predict(self, entrada):
        return self.forward(entrada)[-1][1]

    def train(self, entrada, salida, learning_rate, epochs):


        loss=[]

        # Entrenamos la red i epocas
        for i in range(epochs):
            out = self.forward(entrada)
            # backward
            self.backpropagation(out, salida, learning_rate)
            predicted =  out[-1][1]
            if i % 25 == 0: 
                loss.append(self.loss(predicted, salida))
        return loss

    def test(self, entrada, salida):
        contador = 0
        for index, d in enumerate(entrada):
            predicted_value = self.predict(d)
            if np.rint(predicted_value)[0] == salida[index][0]:
                contador += 1
        return contador / len(entrada)
