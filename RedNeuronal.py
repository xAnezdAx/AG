import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Algoritmo_Genetico import GeneticAlgorit as ga

data = pd.read_csv(
    'D:/UNAL/Semestre 7/Sistemas sinteligentes computacionales/1 Todo lo necesario entrega de proyecto/caesarian.csv')
"""print(data.head())
print(data.shape)
print(data.describe())"""

# reconfiguramos la estructura de la data ya que viene de un dataframe

# Yreal es la variable de salida en el dataset
Y = data['cesaria'].values.reshape(-1, 1)

# X es la variable de entrada pero podrian ser varias
X = data[['edad', 'numero', 'tiempo', 'presion','cardiaco']].values.reshape(-1, 5)

# en una columna
print('Y = ', Y.shape)
print('X = ', X.shape)
print('Y = ', Y)
print('X = ', X)
# Pausa
# input("Presiona Enter para continuar...")

# sigmoid function


def nonlin(x, deriv=False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# entradas, capa oculta
# syn0 = 2*np.random.random((5, 2)) - 1
# capa oculta, salida
# syn1 = 2*np.random.random((2, 1)) - 1


# input("Presiona Enter para continuar...")

eta = 0.4
iterracion = []
vecerror = []


def calcular_pesos(arquitectura):
    # Inicializar la cantidad total de pesos
    total_pesos = 0

    # Recorrer la arquitectura para calcular los pesos
    for i in range(1, len(arquitectura)):
        capa_actual = arquitectura[i]
        capa_anterior = arquitectura[i - 1]

        # Sumar los pesos de la capa actual (por cada conexión)
        total_pesos += capa_actual * capa_anterior

    return total_pesos

def generar_variables_pesos(arquitectura):
    # Generar un string con la forma "x1, x2, x3, ..."
    variables_pesos = ','.join(
        [f"x{i}" for i in range(1, calcular_pesos(arquitectura) + 1)])
    return variables_pesos

def fila_aleatoria(matriz1, matriz2):
    # Obtener un índice aleatorio dentro del rango de filas de la matriz
    indice_aleatorio = np.random.randint(matriz1.shape[0])

    # Seleccionar la fila correspondiente al índice aleatorio
    fila_seleccionada1 = matriz1[indice_aleatorio, :]
    fila_seleccionada2 = matriz2[indice_aleatorio, :]

    return fila_seleccionada1, fila_seleccionada2

def reemplazar_valores(vector, expresion):
    # Asociar valores a las variables
    a, b, c, d, f = vector

    # Reemplazar los valores en la expresión
    expresion_reemplazada = expresion.replace('a', str(a)).replace(
        'b', str(b)).replace('c', str(c)).replace('d', str(d)).replace('f', str(f))

    return expresion_reemplazada

while True:
    restriccion = input("¿Activar AG? (S/N): ")
    if restriccion == 'S' or restriccion == 's':
        restriccion = True
        break
    elif restriccion == 'N' or restriccion == 'n':
        restriccion = False
        break
    else:
        print("Opción inválida, intente de nuevo.")

if restriccion:
    arquitectura = [5, 2, 1]

    variables = generar_variables_pesos(arquitectura)

    fila_aleatoria_1, fila_aleatoria_2 = fila_aleatoria(X, Y)

    n = 5
    Pcruce = 0.8
    Pmuta = 0.2
    generaciones = 5
    restriccion = False
    xRestringe = None
    # funcion objetivo
    expresion = " - (1 / (1 + exp(-((1 / (1 + exp(-(a*x1 + b*x3 + c*x5 + d*x7 + f*x9)))) * x11 + (1 / (1 + exp(-(a*x2 + b*x4 + c*x6 + d*x8 + f*x10)))) * x12))))"
    xObjetivo = str(fila_aleatoria_2[0]) + \
        str(reemplazar_valores(fila_aleatoria_1, expresion))
    print('xObjetivo:', xObjetivo)
    infVariables = {}
    for variable in variables.split(','):
        valMin = float(0)
        valMax = float(2)
        decimales = int(0)
        infVariables[variable] = [valMax, valMin, decimales]
    elitismo = True
    activarConvergencia = False
    valConvergencia = 0.001
    acMaxMin = True
    maximizar = False
    tipoDeSeleccion = "torneo"
    
    instancia = ga(n, Pcruce, Pmuta, generaciones, xRestringe, xObjetivo, infVariables,
                elitismo, activarConvergencia, valConvergencia, restriccion,
                acMaxMin, maximizar, tipoDeSeleccion,methodoCruce='basico')
    instancia.ejecutar()
    # Obtener los valores del diccionario
    # valores = list(ga.get_mejoresIndividuos().values())
    valores = np.array(list(instancia.get_mejoresIndividuos().values()))

    # Convertir los valores en una matriz
    # matriz = np.array(valores)

    # arreglo de pesos
    print('matriz')
    print(valores)

    # Convertir la matriz a tipo flotante
    # matriz = valores.astype(float)

    # Resto del código (reshape y asignación a syn0 y syn1)
    shape_syn0 = (5, 2)
    shape_syn1 = (2, 1)

    syn0 = valores[:np.prod(shape_syn0)].reshape(shape_syn0)
    syn1 = valores[np.prod(shape_syn0):].reshape(shape_syn1)

    print(syn0)
    print(syn1)
else:
    # entradas, capa oculta
    syn0 = 2*np.random.random((5, 2)) - 1
    # capa oculta, salida
    syn1 = 2*np.random.random((2, 1)) - 1
    
"""
print('n:', n)
print('Pcruce:', Pcruce)
print('Pmuta:', Pmuta)
print('generaciones:', generaciones)
print('xRestringe:', xRestringe)
print('xObjetivo:', xObjetivo)
print('infVariables:', infVariables)
print('elitismo:', elitismo)
print('activarConvergencia:', activarConvergencia)
print('valConvergencia:', valConvergencia)
print('restriccion:', restriccion)
print('acMaxMin:', acMaxMin)
print('maximizar:', maximizar)
print('tipoDeSeleccion:', tipoDeSeleccion)
#pausa
input("Presiona Enter para continuar...")"""




for iter in range(5):

    iterracion.append(iter)

    # Propagacion hacia adelante - desde capa 0 (entrada) a capa 1 a capa 2 (salida)
    neta1 = np.dot(X, syn0)
    # print('neta1:', neta1, '\n',)
    l1 = nonlin(neta1)
    # print('salida intermedia l1:', '\n', l1)

    neta2 = np.dot(l1, syn1)
    # print('neta2:', '\n', neta2)
    l2 = nonlin(neta2)
    # print('Salida red:', '\n', l2)
    # input("Presiona Enter para continuar...")
    l2_error = Y - l2

    # ENTRENAMIENTO
    """if (iter % 1000) == 0:
        print('Error Promedio Absoluto:' + str(np.mean(np.abs(l2_error))))
    print('Error:' + str(np.mean(np.abs(l2_error)   )))        
    print('Error:'+  str(np.mean(np.abs(l2_error)**2)))"""
    vecerror.append(np.mean((np.abs(l2_error)**2)))
    # print('error:', vecerror)
    print('Error-l2:', l2_error)

    # Pausa
    # input("pausa Error-l2")

    print('\n', '***********empieza')
    l2_delta = l2_error * nonlin(l2, deriv=True)*eta
    print('l2delta', l2_delta)

    l1_error = l2_delta.dot(syn1.T)
    # print('l1_error', l1_error)
    l1_delta = l1_error * nonlin(l1, deriv=True)*eta
    print('l1_delta', l1_delta)

    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
    print('pesos modificados')
    print(syn0)
    print(syn1)

print('*********************************')
print('Output After Training:')
print('Salida red:', '\n', l2)
print('Error:' + str(np.mean(np.abs(l2_error))), '\n')
print('pesos: ')
print(syn1, '\n')
print(syn0, '\n')
# Con formateo
# print "%5d%10s" %(1,'a')

# Grafica
plt.title('Error vs Iteracion')
plt.plot(iterracion, vecerror)
plt.xlabel('Iteracion')
plt.ylabel('Error')
plt.show()
