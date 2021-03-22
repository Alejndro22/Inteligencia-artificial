import numpy as np
import matplotlib.pyplot as plt
from math import e
import pandas as pd

matriz = pd.read_csv("Prueba.csv", sep=';', comment='#').values
matriz2 = pd.read_csv("Pesos.csv", sep=';', comment='#').values

# Para las graficas, ERROR DE EPOCA
epoch = np.array([])

print("----------- MATRIZ IMPORTADA -----------")
print(matriz)
print("----------- MATRIZ 2 IMPORTADA -----------")
print(matriz2)
num_rows_matriz = np.shape(matriz)[0]
num_rows_matriz2 = np.shape(matriz2)[0]
p = matriz[0:num_rows_matriz, 0:2].copy()
print("----------- MATRIZ FILTRADA DE PATRONES -----------")
print(p)
y = matriz[0:num_rows_matriz, 2:4].copy()
print("----------- MATRIZ FILTRADA DE SALIDAS -----------")
print(y)
# Matriz de pesos para la capa 0
# Quedando de la siguiente manera [ w1,1  w1,2
#                                   w2,1  w2,2 ]
w0_0 = matriz2[0:2, 0:1].copy()
w0 = matriz2[0:2, 1:3].copy()

# Se almacenan los resultados de los patrones
t = np.array([])
# Matriz de pesos para la capa 1
# Quedando de la siguiente manera [ w1,1  w1,2
#                                   w2,1  w2,2 ]
w1_0 = matriz2[2:4, 0:1].copy()
w1 = matriz2[2:4, 1:3].copy()


# Indica cuantas filas tiene la matriz p, para saber cuantos patrones hay
num_rows = np.shape(p)[0]


# Función para hallar la función de transferencia en base a la energía de la nuerona
# Se usa la función logaritmico simoidal
def get_ft(en):
    elev = e ** (-en)
    return 1 / (1 + elev)


# Función para hallar el error
def get_error(y_esp, h):
    return y_esp - h


# Función para hallar el gradiente
def get_grad(err, h):
    print("------------- LOS GRADIENTES DE ESTA CAPA SON -----------------")
    return err * ((h ** 2) - h)


# Función para hallar el gradiente
def get_grad_ant(grad, w, h):
    print("------------- LOS GRADIENTES DE LA CAPA ANTERIOR SON -----------------")
    gpw = np.dot(grad, w)
    return gpw * (-(h ** 2) + h)


# Función para hallar el delta de los pesos
def get_deltapesos(gradientes, hipotesis_1):
    deltapesos = np.array(np.dot(gradientes.transpose(), hipotesis_1))
    return - deltapesos


# flag = True
# while flag:
#     flag = False
ciclos = 500
for aa in range(ciclos):
    errorEpoca = 0
    for li in range(num_rows):
        # Mostrar que patrón se analiza
        print("     ANALIZANDO FILA " + str(li))
        # MOSTRAR LOS PESOS Y COMO VAN CAMBIANDO
        print("----------- MATRIZ FILTRADA WK0 -----------")
        print(w0)
        print("----------- MATRIZ FILTRADA WK0_0 -----------")
        print(w0_0)
        print("----------- MATRIZ FILTRADA WK1 -----------")
        print(w1)
        print("----------- MATRIZ FILTRADA WK1_0 -----------")
        print(w1_0)
        # PROCESO PARA OBTENER LOS ERRORES
        # La neurona de la capa 0 ya tiene sus patrones y pesos, procedo a encontrar energías de las neuronas en k0
        print("----- CAPA 0 -----")
        row_try = (p[li, :])
        print(row_try)
        e0 = (np.dot(row_try, w0.transpose())) + w0_0.transpose()
        print("-- Las energías quedan v1 --")
        print(e0)
        # Luego se obtienen las h que serán las entradas para la capa 1
        print("-- Las h quedan --")
        h0 = get_ft(e0)
        print(h0)

        # Ahora la capa 1
        # Se obtienen las energías
        print("----- CAPA 1 -----")
        e1 = np.dot(h0, w1.transpose()) + w1_0.transpose()
        print("-- Las energías quedan --")
        print(e1)
        # Luego se obtienen las h que serán las salidas de la capa 1
        print("-- Las h quedan --")
        h1 = get_ft(e1)
        print(h1)
        error = get_error((y[li, :]), h1)
        print("-- El vector de errores es --")
        print(error)
        errorT = (np.sum(error ** 2))/2
        print("---- ERROR TOTAL ----")
        print(errorT)
        gradk1 = np.array(get_grad(np.sum(error), h1))
        print(gradk1)
        delta = get_deltapesos(gradk1, h0)
        print("-- El vector de deltas w1 y w2 es --")
        print(delta)
        nuevoW = delta
        nuevoW_0 = gradk1.transpose() * (-1)
        print("-- El vector delta w0 es --")
        print(nuevoW_0)
        gradk0 = get_grad_ant(gradk1, w1, h0)
        print(gradk0)
        delta = get_deltapesos(gradk0, np.array([row_try]))
        print("-- El vector de deltas para la capa anterior w1 y w2 es --")
        print(delta)
        w1 = w1 + nuevoW
        w1_0 = w1_0 + nuevoW_0
        nuevoW_0 = gradk0.transpose() * (-1)
        print("-- El vector delta w0 es --")
        print(nuevoW_0)
        w0 = w0 + delta
        w0_0 = w0_0 + nuevoW_0
        print()
        errorEpoca = (errorEpoca + errorT)

    epoch = np.append(epoch, errorEpoca/num_rows)

print("-------- AAAAAAAAAAAAAAAAa ------")
print(epoch)
print("-------- BBBBBBBBBBBBBBBBb ------")
arrayHorizontal = np.linspace(0, ciclos-1, ciclos)
print(arrayHorizontal)
plt.plot(arrayHorizontal, epoch)
plt.show()
