import numpy as np
from math import e

# Para comenzar pruebas, definiré los patrones de entrada y salida esperada desde acá
#              p1 p2
p = np.array([[1, 0],
              [0, 1],
              [1, 1]])
#              y1 y2
y = np.array([[1, 0],
              [0, 1],
              [1, 0]])
# Matriz de pesos aleatorios para la capa 0 de valores entre 0-1
# Quedando de la siguiente manera [ w1,1  w1,2
#                                   w2,1  w2,2 ]
w0 = np.random.rand(*(2, 2))
# Vector de energías para capa 0
e0 = np.array([0, 0])
# Vector de h para capa 0
h0 = np.array([0, 0])

# Matriz de pesos aleatorios para la capa 1 de valores entre 0-1
# Quedando de la siguiente manera [ w1,1  w1,2
#                                   w2,1  w2,2 ]
w1 = np.random.rand(*(2, 2))
# Vector de energías para capa 1
e1 = np.array([0, 0])
# Vector de h para capa 1
h1 = np.array([0, 0])

# Indica cuantas filas tiene la matriz p, para saber cuantos patrones hay
num_rows = np.shape(p)[0]


# Función para hallar la función de transferencia en base a la energía de la nuerona
# Se usa la función logaritmico simoidal
def get_ft(en):
    elev = e ** (-en)
    return 1 / (1 + elev)


# PROCESO PARA OBTENER LOS ERRORES
# La neurona de la capa 0 ya tiene sus patrones y pesos, procedo a encontrar energías de las neuronas en k0
print("----- CAPA 0 -----")
row_try = (p[2, :])
print(row_try)
print("-- w0 --")
print(w0)
print("-- w0 transpuesta --")
print(w0.transpose())
e0 = np.dot(row_try, w0.transpose())
print("-- Las energías quedan --")
print(e0)
# Luego se obtienen las h que serán las entradas para la capa 0
print("-- Las h quedan --")
h0 = get_ft(e0)
print(h0)

# Ahora la capa 1
# Se obtienen las energías
print("----- CAPA 1 -----")
print("-- w1 --")
print(w1)
print("-- w1 transpuesta --")
print(w1.transpose())
e1 = np.dot(h0, w1.transpose())
print("-- Las energías quedan --")
print(e1)
# Luego se obtienen las h que serán las entradas para la capa 1
print("-- Las h quedan --")
h1 = get_ft(e1)
print(h1)
