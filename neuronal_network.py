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

    def train(self, entrada, salida, learning_rate):


        loss=[]

        for i in range(2000):
            #Entrenemos la red!
            out = self.forward(entrada)
            # backward
            deltas = self.backpropagation(out, salida, learning_rate)
            pY =  out[-1][1]

            if i % 25 ==0: 
                #print(pY) 
                loss.append(self.loss(pY, salida))
                res = 50
                _x0 = np.linspace(-1.5, 1.5, res)
                _x1 = np.linspace(-1.5, 1.5, res)
                
                _Y = np.zeros((res, res))

                for i0, x0 in enumerate(_x0):
                    for i1, x1 in enumerate(_x1):
                        _Y[i0, i1] = self.predict(np.array([[x0, x1]]))[0][0] 

                plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
                plt.axis("equal")
                plt.scatter(entrada[salida[:,0]==0, 0] , entrada[salida[:,0]==0, 1], c="skyblue")
                plt.scatter(entrada[salida[:,0]==1, 0] , entrada[salida[:,0]==1, 1], c="salmon")
                clear_output(wait=True)
                plt.show(block=False)
                plt.pause(0.01)
                plt.close()
                plt.plot(range(len(loss)), loss)
                plt.show(block=False)
                plt.pause(0.01)
                plt.close()
                print(_Y)

