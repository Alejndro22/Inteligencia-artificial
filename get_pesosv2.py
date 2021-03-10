import numpy as np
import matplotlib.pyplot as plt

w = np.array([[0.5, -0.7, 0.2]])
p = np.array([[1, 1, 1, 1],
              [2, 0, -2, 0],
              [1, -1, 1, 2]])
t = np.array([[1],
              [1],
              [0],
              [0]])


def get_y2(x):
    return (-x * w.item(1) - w.item(0)) / w.item(2)


def neurona_out(W, P):
    ope1 = np.dot(W, P)
    print("El valor es")
    print(ope1)
    if ope1 < 0:
        return 0
    else:
        return 1


def graph():
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.plot(2, 1, 'bo', color='green')
    plt.plot(0, -1, 'bo', color='green')
    plt.plot(-2, 1, 'bo', color='red')
    plt.plot(0, 2, 'bo', color='red')
    plt.axis(ymin=-2, ymax=3)


def get_y_orientation(y):
    if y > 0:
        return 500
    else:
        return -500


x_limites = np.linspace(-5, 5, 20)
# bandera para saber si los pesos no cambiaron
flag = True
# Indica cuantas columnas tiene la matriz
num_columns = np.shape(p)[1]
# Variable que lleva la cuenta de las iteraciones
cont = 1
# muestra la primer grafica con pesos iniciales
graph()
y2 = get_y2(x_limites)
plt.fill_between(x_limites, y2, -5, alpha=0.10, color='b')
plt.show()

while flag:
    print("------- Iteración número #" + str(cont) + " -------")
    # Comienza la bandera como falsa
    flag = False
    print("----- W es -----")
    print(w)
    for li in range(num_columns):
        ope = neurona_out(w, p[:, [li]])
        print("Salida neurona p" + str(li))
        print(ope)
        print("error")
        print(t.item(li) - ope)
        if (t.item(li) - ope) != 0:
            w = (-2 * (t.item(li) - np.dot(p[:, [li]], w)))
            w = np.transpose(np.dot(w, p[:, [li]]))
            # w = w + (t.item(li) - ope) * p[:, [li]].transpose()
            print("Nuevos pesos")
            print(w)
            flag = True
            graph()
            y2 = get_y2(x_limites)
            plt.fill_between(x_limites, y2, get_y_orientation(w.item(2)), alpha=0.10, color='b')
            plt.show()

        print("La bandera sale")
        print(flag)
    cont += 1
    print("")
