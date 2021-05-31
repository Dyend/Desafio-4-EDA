import neuronal_network
import sys
import os
from activation import Sigmoid
from loss import MSE
from util import plot_loss, read_data, pre_proc, split_data, save_nn, load_nn

error_path = "out/error.png"
data_path = "data/dataset.csv"
nn_path = "data/neural_network.dat"


def print_menu():
    print('1- Entrenar red neuronal desde 0')
    print('2- Cargar red neuronal')
    print('3- Guardar red neuronal')
    print('4- Testear red neuronal')
    print('Presione cualquier tecla para salir')

def opcion_1(data):
    training, test = pre_proc(data, 70)
    entrada, salida = split_data(training, salida_columns)
    entrada_test, salida_test = split_data(test, salida_columns)

    topology = [entrada[0].size, 8, 4, salida[0].size]
    epochs = 500
    learning_rate = 0.1
    """Topology es una lista con la cantida de neuronas en cada capa
        act_f es la funcion de activacion que sera usada por cada capa """
    print('Creando red neuronal de topologia: ', topology)
    nn = neuronal_network.Network(topology, Sigmoid, MSE)
    print('Iniciando entrenamiento...')
    error = nn.train(entrada, salida, learning_rate, epochs)
    plot_loss(error, error_path)
    print('Entrenamiento finalizado.')

    opcion_4(nn, entrada_test, salida_test)

    return nn, entrada_test, salida_test

def opcion_2():
    return load_nn(nn_path)

def opcion_3(nn):
    save_nn(nn, nn_path)

def opcion_4(nn, entrada_test, salida_test):
    print('Iniciando testing...')
    accuracity = nn.test(entrada_test, salida_test)
    print('Accuracity: ', accuracity)



def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')

if __name__ == '__main__':
    """ se lee el dataset y se pre procesa"""
    unwanted_cols = ['song_title', 'artist', 'duration_ms', 'key', 'mode', 'popularity', 'time_signature']
    salida_columns = [-1]
    data, headers = read_data(data_path, unwanted_cols)
    print(headers)
    nn, entrada_test, salida_test = opcion_1(data)

    while True:
        input('Enter para continuar...')
        screen_clear()
        print_menu()
        opcion = input()
        if opcion == "1":
            nn, entrada_test, salida_test = opcion_1(data)
        elif opcion == "2":
            nn = opcion_2()
        elif opcion == "3":
            opcion_3(nn)
        elif opcion == "4":
            opcion_4(nn, entrada_test, salida_test)
        else:
            sys.exit(0)



