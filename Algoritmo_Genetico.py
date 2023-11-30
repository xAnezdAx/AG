import math
import random
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import sympy as sp #libreria para operaciones simbolica
from matplotlib.animation import FuncAnimation

from IPython.display import display, clear_output

import sys
import os
import time
from datetime import datetime

class GeneticAlgorit:
    def __init__(self, nIndiv, Pcruce, Pmuta, generaciones, restricciones, exprObjetivo, 
                infVariables, elitismo, activarConvergencia, valConvergencia, restriccionIngreso,
                acMaxMin,maximizar,selccion,methodoCruce=None):
        
        self.selccion = selccion #tipo de seleccion de padres
        self.nIndiv = nIndiv #numero de individuos
        self.Pcruce = Pcruce #probabilidad de cruce
        self.Pmuta = Pmuta   #probabilidad de mutacion
        self.generaciones = generaciones #numero de generaciones
        self.restricciones = restricciones #restricciones tantas como sean necesarias
        self.exprObjetivo = exprObjetivo   #funcion objetivo
        self.infVariables = infVariables   #variables de decision  infVariables -> diccionario valMax valMin decimales
        self.acMaxMin = acMaxMin #activar maximizacion o minimizacion
        self.maximizar = maximizar #maximizar o minimizar
        self.elitismo = elitismo      #activar elitismo
        self.converge = valConvergencia #porcentaje maximo de convergencia
        self.activarConvergencia = activarConvergencia #activar convergencia
        self.activarRestriccion = restriccionIngreso #activar restricciones
        self.bitGen = {} #cantidad de bit que usa cada gen o variable de decision del individuo
        self.totalBits = 0 #total de bits que usa el individuo
        self.fitness = np.empty((self.nIndiv))
        self.acumulado = np.empty((self.nIndiv))
        self.pesosCalculados = np.empty((self.nIndiv), dtype=bool)#evaluacion de factibilidad
        self.resumen = []
        self.valorDesplazamiento=0
        self.mejoresVariablesDesicion = []
        self.methodoCruce=methodoCruce

    def mejorIndividuo(self,n,fitnessPoblIterada,poblItlocal,maximizar):
        #buscar el mejor individuo de la poblacion iterada
        mejor=0
        if maximizar:   #buscar el fit mas grande         
            for i in range(0, n):
                if fitnessPoblIterada[i]>fitnessPoblIterada[mejor]:
                    mejor=i
        else:            #buscar el fit mas pequeño
            for i in range(0, n):
                if fitnessPoblIterada[i]<fitnessPoblIterada[mejor]:
                    mejor=i
        return poblItlocal[mejor] , fitnessPoblIterada[mejor]   

    def ejecutar(self):
        #inicializar variables
        
        self.listaDeBits()
        self.mejoresIndividuos = [] #individuo binario - fitness
        #crear poblacion inicial binaria aleatoria y verificada               
        self.poblIt = self.crearPoblacionInicial(self.nIndiv, self.totalBits)
        total,fitnessDesplazados,variable_values=self.evalua(self.poblIt)
        self.imprime(self.nIndiv,total,self.fitness,self.poblIt,self.pesosCalculados,fitnessDesplazados,variable_values)
        print("##################################################################")

        ejex = np.array([])
        ejey = np.array([])
        ejez = np.array([])

        plt.ion()  # Habilita el modo interactivo
        fig, ax = plt.subplots()

        for iter in range(self.generaciones):
            print("\n Iteracion ", iter, "\n")
            print("Poblacion de la Iteracion ", iter, "\n", self.poblIt)
            mejorInd, mejorFit = self.mejorIndividuo(self.nIndiv,self.fitness,self.poblIt,self.maximizar)
            print("El mejor individuo es:", mejorInd)
            print("El mejor fitness es:", mejorFit)
            #aplicar elitimos
            if self.elitismo:
                hijosTemporales = np.array([mejorInd])
            else:
                hijosTemporales = np.array([])
            
            
            #print("hijosTemporales sin iterar: ", hijosTemporales)
            #elitimos
            while len(hijosTemporales) < self.nIndiv:
                #print("hijosTemporales: ", hijosTemporales)
                hijoA, hijoB = self.obtener_hijos()
                #concatenar dos arrays de NumPy que no tienen las mismas dimensiones no esta permitido
                if self.factible(hijoA) and len(hijosTemporales) < self.nIndiv:
                    print("hijoA: ", hijoA)
                    if hijosTemporales.size == 0:
                        hijosTemporales = np.array([hijoA])
                    else:
                        hijosTemporales = np.vstack((hijosTemporales,hijoA))
                else:
                    print("hijoA no factible: ", hijoA)

                if self.factible(hijoB) and len(hijosTemporales) < self.nIndiv:
                    print("hijoB: ", hijoB)
                    if hijosTemporales.size == 0:
                        hijosTemporales = np.array([hijoB])
                    else:
                        hijosTemporales = np.vstack((hijosTemporales,hijoB))
                else:
                    print("hijoB no factible: ", hijoB)

            self.poblIt = hijosTemporales.copy()
            total,fitnessDesplazados,variable_values=self.evalua(self.poblIt)
            mejorIndNew, mejorFitNew = self.mejorIndividuo(self.nIndiv,self.fitness,
                                                        self.poblIt,self.maximizar)
            self.mejoresIndividuos.append([mejorIndNew, mejorFitNew])
            
            offline=0
            offline=(sum(item[1] for item in self.mejoresIndividuos))/len(self.mejoresIndividuos)
            #         iteracion-Curva best-so-far - Curva off-line - Curva online
            self.resumen.append([iter, mejorFit, offline, total/self.nIndiv])           
            print("\nnueva poblacion al terminar iteracion:", iter, "\n", self.poblIt)
            self.imprime(self.nIndiv,total,self.fitness,self.poblIt,self.pesosCalculados,fitnessDesplazados,variable_values)
            ejex = np.append(ejex, self.resumen[iter][1])
            ejez = np.append(ejez, self.resumen[iter][2])
            ejey = np.append(ejey, self.resumen[iter][3])

            ax.clear()  # Limpia el gráfico para actualizar los datos
            ax.plot(ejex, label='The best Z (best-so-far)', color='green')
            ax.plot(ejez, label='Average best Z (off-line)', color='blue')
            ax.plot(ejey, label='Average Z (online)', color='red')
            ax.set_ylabel('Valor de Z')
            ax.set_xlabel('Iteraciones')
            ax.set_title('Z con el pasar de las iteraciones')
            ax.legend()
            plt.pause(0.01)
            #evaluamos si puede seguir iterando
            if self.activarConvergencia:
                if self.convergencia(self.fitness,self.converge):
                    print("Convergencia alcanzada")
                    print("Iteraciones fijalizadas en la iteracion: ", iter)
                    break
            

        plt.ioff()
        self.graficar()
        plt.show()
            
    def obtener_hijos(self):#padres, mutacion, cruce, hijos
        
        x=self.totalBits
        papa1 = self.seleccion()
        papa2 = self.seleccion()
        hijoA, hijoB = self.cruce(np.random.rand(), papa1, papa2)
        table1 = PrettyTable(['hijoA antes de mutar', 'hijoB antes de mutar'])
        table1.add_row([hijoA, hijoB])
        print(table1)
        for a in range(x):
            if np.random.rand() < self.Pmuta:
                hijoA[a] = (int)(1 - hijoA[a])
            if np.random.rand() < self.Pmuta:
                hijoB[a] = (int)(1 - hijoB[a])
        table = PrettyTable(['hijoA despues de mutar', 'hijoB despues de mutar'])
        table.add_row([hijoA, hijoB])
        print(table)
        return hijoA, hijoB
    
    def seleccion(self): #seleccion de padres
        print("\nseleccion de padres por: ",self.selccion)
        if self.selccion == "ruleta":
            escoje1 = np.random.rand()
            for i in range(0, self.nIndiv):
                if self.acumulado[i] > escoje1:
                    padre = self.poblIt[i]
                    break
        elif self.selccion == "torneo":
            escoje1 = np.random.randint(0, self.nIndiv)
            escoje2 = np.random.randint(0, self.nIndiv)
            if self.maximizar:
                if self.fitness[escoje1] > self.fitness[escoje2]:
                    padre = self.poblIt[escoje1]
                else:
                    padre = self.poblIt[escoje2]
            else:
                if self.fitness[escoje1] < self.fitness[escoje2]:
                    padre = self.poblIt[escoje1]
                else:
                    padre = self.poblIt[escoje2]
        return padre
        
    def cruce(self,a1,p1,p2):
        print("\npadres sin cruzarsen entre si:\n")
        print("Padre 1: ", p1)
        print("Padre 2: ", p2)
        if a1<self.Pcruce:
            print("\nMas grande", self.Pcruce, "que ", a1, "-> Si Cruzan\n")
            
            if len(p1) != len(p2):
                print("Error: Parents must be of the same length for crossover.")
                return p1, p2

            if self.methodoCruce == 'Segmentado':
                # Segment crossover
                segment = random.randint(0, len(p1) - 1)
                hijo1 = np.concatenate((p1[:segment], p2[segment:]), axis=None)
                hijo2 = np.concatenate((p2[:segment], p1[segment:]), axis=None)

            elif self.methodoCruce == 'Multipunto':
                # Multipoint crossover
                points = sorted(random.sample(range(len(p1)), 2))
                length = min(points[1] - points[0], len(p2) - points[0])  # Asegurar que la longitud no exceda la del padre más corto
                temp1 = np.concatenate((p1[:points[0]], p2[points[0]:points[0]+length], p1[points[0]+length:]), axis=None)
                temp2 = np.concatenate((p2[:points[0]], p1[points[0]:points[0]+length], p2[points[0]+length:]), axis=None)

                hijo1 = np.concatenate((temp1[:points[0]], temp2[points[0]:points[0]+length], temp1[points[0]+length:]), axis=None)
                hijo2 = np.concatenate((temp2[:points[0]], temp1[points[0]:points[0]+length], temp2[points[0]+length:]), axis=None)

                print("\nhijos después de cruzarse entre sí:\n")
                print("Hijo 1: ", hijo1)
                print("Hijo 2: ", hijo2)


            elif self.methodoCruce == 'Uniforme':
                # Uniform crossover
                hijo1, hijo2 = [], []
                for i in range(len(p1)):
                    if random.random() < 0.5:
                        hijo1.append(p1[i])
                        hijo2.append(p2[i])
                    else:
                        hijo1.append(p2[i])
                        hijo2.append(p1[i])
            elif self.methodoCruce == 'Basico':
                num_decimales = 2 
                porcentajesindividual = 1 / (len(p1) - 1)
                corte = 1
                para_corte = np.empty((len(p1) - 1))
                para_corte[0] = round(porcentajesindividual, num_decimales) 
                porcentajes=porcentajesindividual
                for i in range(1, len(para_corte)):
                    para_corte[i] = round(porcentajes+para_corte[i-1], num_decimales)
                while a1>porcentajes:
                    corte+=1
                    porcentajes+=porcentajesindividual            
                temp1=p1[0:corte] #[i:j] corta desde [i a j)
                temp2=p1[corte:len(p1)]
                print(temp1,temp2)

                temp3=p2[0:corte]
                temp4=p2[corte:len(p2)]
                print(temp3,temp4)

                hijo1 = list(temp1)
                hijo1.extend(list(temp4))

                hijo2 = list(temp3)
                hijo2.extend(list(temp2))
            
            print("\nhijos despues de cruzarse entre si:\n")
            print("Hijo 1: ", hijo1)
            print("Hijo 2: ", hijo2)

        else:
            print("\nMenor", self.Pcruce, "que ", a1, "-> NO Cruzan\n")
            hijo1=p1
            hijo2=p2
    
        return hijo1,hijo2

    def imprime(self, n,total,fitness,poblIt,pesosCalculados,fitnessDesplazados,variable_values):
        
        print ("\n",'Tabla Iteracion:',"\n")
        # Creamos la tabla con las columnas 'Individuo', 'Cromosoma', 'Fitness', 'Probabilidad', 'Acumulada' y 'restriccion'
        print("valor de desplzamiento como tratamiento de negativos, constante de desplazamiento 1")
        print("valor de desplazamiento: ",self.valorDesplazamiento)
        table = PrettyTable(['Individuo', 'Cromosoma','valor variables', 'Fitness','Fitness desplazado', 'Probabilidad', 'Acumulada', 'restriccion'])
        # Iteramos sobre los datos y agregamos cada fila a la tabla
        acumula = 0 
        for i in range(0, n):
            if self.acMaxMin:#debe estar activa la minimizacion o maximizar 
                if self.maximizar:
                    probab = fitnessDesplazados[i] / total
                    acumula += probab
                elif not self.maximizar: #pendiente correguir
                    """if (1 - (fitnessDesplazados[i])) < 0:
                        procesar = 0
                    else:
                        procesar = (1 - (fitnessDesplazados[i]))"""
                    probab =  fitnessDesplazados[i] / total
                    acumula += probab
            else:
                probab = fitnessDesplazados[i] / total
                acumula += probab
            #probab = fitness[i] / total
            #acumula += probab
            self.acumulado[i]=acumula
            table.add_row([i + 1, poblIt[i], variable_values[i], fitness[i],
                        fitnessDesplazados[i], "{0:.3f}".format(probab), 
                        "{0:.3f}".format(acumula), pesosCalculados[i]])
        #mejores variables de desicion
        
        self.mejoresVariablesDesicion.append(variable_values[-1])
        # Agregamos la fila con el total de fitness
        #table.add_row(['', '', '', '', '', '', ''])
        #table.add_row(['Total Fitness:', '', '', total, '', '', ''])
        print(table)
        #return self.acumulado

    def obtener_fitness(self,individ):
        #print("\n","Individuo a evaluar:","\n", individ)
        variable_values = {}
        inicio = 0
        cantidad=0
        try:
            # Convertir la cadena de entrada en una expresión simbólica
            print("Expresion objetivo: ", self.exprObjetivo)
            #expr = sp.sympify(self.exprObjetivo)
            expr =  sp.sympify(self.exprObjetivo, evaluate=False)
            print("Expresion objetivo transformada: ", expr)
            # Solicitar valores para las variables
            claves = self.infVariables.keys()
            for var in claves:
                cantidad += self.bitGen[var]
                binario = ''
                for i in range(inicio,cantidad):
                    binario += str(individ[i])
                    inicio += 1
                #inicio += 1
                #print("valor de binario: ",binario)
                var_value = self.conversionBinarioDecimal(self.infVariables[var][1],self.infVariables[var][0],self.bitGen[var],binario,self.infVariables[var][2])
                variable_values[var] = var_value
            print("valor de variable_values: ",variable_values)
            # Evaluar la expresión con los valores ingresados especifica para verdad o falso
            result = expr.subs(variable_values).evalf()
            
            #print(variable_values)
            print(f"Resultado de la evaluación de fitness: {result}")
            return result
        except (sp.SympifyError, ValueError) as e:
            print("Error al procesar la expresión:", e)        

    def evalua(self,poblIt):
        #print("\n","Poblacion a evaluar:","\n", poblIt)
        #x=len(self.infVariables)        
        claves = self.infVariables.keys()
        acumula = 0
        n = self.nIndiv
        variable_values = []
        for i in range(0, n):
            #se debe obtener el valor decimal de cada variable
            variable_values_temp = {}
            inicio = 0
            cantidad=0
            individ = poblIt[i]
            for var in claves:
                cantidad += self.bitGen[var] #cantidad de bits que usa cada variable de decision
                binario = ''
                #print('valor cantidad: ',cantidad)
                for j in range(inicio,cantidad): #agrupamos los bits que corresponden a cada variable de decision
                    binario += str(individ[j])
                    inicio += 1  #definir donde inicia la siguiente variable de decision
                #inicio += 1
                #print("valor de binario: ",binario)
                variable_values_temp[var] = self.conversionBinarioDecimal(self.infVariables[var][1],
                                                                    self.infVariables[var][0],
                                                                    self.bitGen[var],
                                                                    binario,
                                                                    self.infVariables[var][2])
                variable_values.append(variable_values_temp)
            
        #print("\n","Valores de las variables:","\n", variable_values)
        
        n=self.nIndiv
        total=0
        for i in range(0, n):     
            #print(self.obtener_fitness(poblIt[i]))
            self.fitness[i]=self.obtener_fitness(poblIt[i])#fitnes de un individuo
            #total += self.fitness[i]
            self.pesosCalculados[i]=self.calcularRestriccion(poblIt[i])#si cumple las restricciones
        #return self.fitness,total,self.pesosCalculados#mejora
        
        
    #si la funcion genera valores negativos los trabajamos
        fitnessTemporales = np.empty((self.nIndiv))
        if self.selccion == "ruleta":
            valor_minimo = min(self.fitness)            
            if valor_minimo < 0:      
                self.valorDesplazamiento = abs(valor_minimo) + 1 #se suma 1 para que no quede en 0, constante de desplazamiento
                for i in range(0, self.nIndiv):                    
                    fitnessTemporales[i] = round(self.fitness[i] + self.valorDesplazamiento, 2)
            else:
                self.valorDesplazamiento = 0
                fitnessTemporales = self.fitness.copy()
        total=sum(fitnessTemporales)
        return total, fitnessTemporales, variable_values
    
    def convergencia(self, lista, valorConvergencia):
        conteo_elementos = {}
        total_elementos = len(lista)
        
        for elemento in lista:
            if elemento in conteo_elementos:
                conteo_elementos[elemento] += 1
            else:
                conteo_elementos[elemento] = 1
        
        porcentaje_elementos = {}
        
        for elemento, cantidad in conteo_elementos.items():
            porcentaje = (cantidad / total_elementos) * 100
            porcentaje_elementos[elemento] = porcentaje

        for porcentaje in porcentaje_elementos.values():  # Usamos .values() para obtener solo los valores
            if porcentaje >= valorConvergencia:
                return True
            
        return False

    def listaDeBits(self):#cantidad de bit que usa cada gen del individuo
        claves = self.infVariables.keys()
        for clave in claves:
            lista = self.infVariables[clave]
            superior = lista[0]
            inferior = lista[1]
            decimales = lista[2]
            cantidad =self.calcular_logaritmo_redondear(superior, inferior, decimales)
            self.bitGen[clave] = cantidad
            self.totalBits += cantidad 
        
    def calcular_logaritmo_redondear(self, limitSuper, limitInferior, decimales):
        resultado = (1 + (limitSuper - limitInferior) * math.pow(10, decimales))
        logaritmo = math.log(resultado, 2)
        redondeado = math.ceil(logaritmo)
        return redondeado
    
    def crearPoblacionInicial(self, nIndivIN, totalBitIN):
        # generar poblacion inicial aleatoria y verificada
        poblInicial = np.random.randint(0, 2, (nIndivIN, totalBitIN))
        print("Poblacion inicial Aleatoria:","\n", poblInicial)
        for fila in range(nIndivIN):
            if not self.calcularRestriccion(poblInicial[fila]):
                while True:
                    poblInicial[fila] = np.random.randint(0, 2, (1, totalBitIN))
                    if self.calcularRestriccion(poblInicial[fila]):
                        break
        print("Poblacion inicial verificada con restricciones:","\n", poblInicial)
        return poblInicial

    def factible(self,individuo):
        return self.calcularRestriccion(individuo)

    def calcularRestriccion(self, hijo):
        variable_values = {}
        inicio = 0
        cantidad=0
        if self.activarRestriccion:
            try:
                # Convertir la cadena de entrada en una expresión simbólica
                #expr = sp.sympify(self.restricciones)
                #provicional
                expr = sp.sympify(self.restricciones, evaluate=False)
                # cargar valores para las variables
                claves = self.infVariables.keys()
                for var in claves:
                    cantidad += self.bitGen[var]
                    binario = ''
                    for i in range(inicio,cantidad):
                        binario += str(hijo[i])
                        inicio += 1
                    #inicio += 1
                    #print("valor de binario: ",binario)
                    var_value = self.conversionBinarioDecimal(self.infVariables[var][1],self.infVariables[var][0],self.bitGen[var],binario,self.infVariables[var][2])
                    variable_values[var] = var_value
                # Evaluar la expresión con los valores ingresados especifica para verdad o falso
                result = expr.subs(variable_values)
                return result
            except (sp.SympifyError, ValueError) as e:
                print("Error al procesar la expresión:", e)
        else:
            return True                                                                  

    def conversionBinarioDecimal(self,min,max,cantBit,gen,decimales):
        binario_decimal = int(gen, 2)
        resultado = min + binario_decimal * (max - min) / (2 ** cantBit - 1)
        resultado = round(resultado, decimales)
        return resultado      

    def graficar(self): 
        iteraciones = [item[0] for item in self.resumen]
        best_so_far = [item[1] for item in self.resumen]
        off_line = [item[2] for item in self.resumen]
        online = [item[3] for item in self.resumen]
        
        plt.figure(figsize=(12, 6))

        # Definir el área de trazado como una cuadrícula de 2 filas y 1 columna
        plt.subplot(3, 1, 1)
        plt.plot(iteraciones, best_so_far, marker='o', linestyle='-', color='b')
        plt.title('The best Z (best-so-far)')
        plt.xlabel('Iteración')
        plt.ylabel('valor')
        plt.grid(True)

        # Continuar trazando en la misma área de trazado
        plt.subplot(3, 1, 2)
        plt.plot(iteraciones, online, marker='o', linestyle='-', color='g')
        plt.title('Average Z (online)')
        plt.xlabel('Iteración')
        plt.ylabel('valor')
        plt.grid(True)

        # Continuar trazando en la misma área de trazado
        plt.subplot(3, 1, 3)
        plt.plot(iteraciones, off_line, marker='o', linestyle='-', color='r')
        plt.title('Average best Z (off-line)')
        plt.xlabel('Iteración')
        plt.ylabel('valor')
        plt.grid(True)

        plt.tight_layout()

        plt.show()    
            
    def get_mejoresIndividuos(self):
        print("\n","Mejores individuos","\n")
        print(self.mejoresVariablesDesicion[-1])
        return self.mejoresVariablesDesicion[-1]
        
        
#Iniciamos el programa

if __name__ == "__main__":
    # Nombre del archivo
    directorio = 'D:/UNAL/Semestre 7/Sistemas sinteligentes computacionales/1 Todo lo necesario entrega de proyecto/Para entregar/AG/ejecuciones'
                
    #formato de tiempo dia_mes_año_hora_minutos
    tiempo_actual = datetime.now().strftime('%d_%m_%Y_%H_%M')
    nombre_archivo = 'salida_' + tiempo_actual + '.txt'
    ruta_completa = os.path.join(directorio, nombre_archivo)

    # Abre el archivo en modo de escritura
    archivo = open(ruta_completa, 'w')
    
    #funcion objetivo
    xObjetivo = input("Ingrese la descripción de la función"\
        "(por ejemplo, 'sin(x)*sin(y)+25' '((x+(2*y)+7)**2 ) + (((2*x)+y-5)**2)': \n")
    variables = input("Ingrese los nombres de las variables separados por comas (por ejemplo, 'x,y,z'): \n")
    #infVariables -> diccionario valMax valMin decimales
    infVariables = {}
    #variables de decisión rango de valores permitidos y sus decimales
    for variable in variables.split(','):
        valMin = float(input(f"Ingrese el valor mínimo de {variable}: "))
        valMax = float(input(f"Ingrese el valor máximo de {variable}: "))
        decimales = int(input(f"Ingrese el número de decimales de {variable}: "))
        infVariables[variable] = [valMax, valMin, decimales]

    #poblacion
    n = int(input("Ingrese el número de individuos en la población (n): "))
    Pcruce = float(input("Ingrese la probabilidad de cruce (Pcruce): "))
    Pmuta = float(input("Ingrese la probabilidad de mutación (Pmuta): "))
    generaciones = int(input("Ingrese el número de generaciones: "))

    #restricciones
    while True:
        restriccion = input("¿Desea agregar restricciones? (S/N): ")
        if restriccion == 'S' or restriccion == 's':
            restriccion = True           
            xRestringe = input("Ingrese la descripción de las restricciones"\
                "(por ejemplo, 'x+y<=10', '(x < 3) | (x > 4)' operador( & | ): \n")
            break
        elif restriccion == 'N' or restriccion == 'n':
            restriccion = False
            xRestringe = None
            break
        else:
            print("Opción inválida, intente de nuevo.")    
    
    #convergencia
    while True:
        activarConvergencia = input("¿Desea utilizar convergencia? (S/N): ")
        if activarConvergencia == 'S' or activarConvergencia == 's':
            activarConvergencia = True
            while True:
                valConvergencia = int(input("Ingrese el porcentaje de convergencia maximo 'entre 0~100': "))
                if valConvergencia >= 0 and valConvergencia <= 100:
                    break
                else:
                    print("Opción inválida, intente de nuevo.")
            break
        elif activarConvergencia == 'N' or activarConvergencia == 'n':
            activarConvergencia = False
            valConvergencia = 0
            break
        else:
            print("Opción inválida, intente de nuevo.")
    
    #activar maximizacion o minimizacion    
    while True:
        print("\nsi la funcion maximizar o minimizar se encuentra desactivada no podra usar elitismo")
        print("si maximiza o minimiza sin elitmismo se almacenaran los mejores z segun corresponda\n")
        acMaxMin = input("¿Desea activar maximizar o minimizar? (S/N):")
        if acMaxMin == 'S' or acMaxMin == 's':
            acMaxMin = True
            break
        elif acMaxMin == 'N' or acMaxMin == 'n':
            acMaxMin = False
            break
        else:
            print("Opción inválida, intente de nuevo.")
    
    #seleccionar maximizar o minimizar
    if acMaxMin:
        while True:
            maximizar = input("si desea maximizar (S), si desea minimizar (N):")
            if maximizar == 'S' or maximizar == 's':
                maximizar = True
                break
            elif maximizar == 'N' or maximizar == 'n':
                maximizar = False
                break
            else:
                print("Opción inválida, intente de nuevo.")
    else:
        maximizar = None
    
    #tipo de seleccion de padres
    while True:
        print("\nseleccione el tipo de clasificacion de padres")
        print("1. Ruleta")
        print("2. Torneo")
        print("3. Ranking")
        print("4. Aleatorio")
        print("5. sorte")
        selccion = int(input("Ingrese el numero de la opcion: "))
        if selccion == 1:
            tipoDeSeleccion="ruleta"
            break
        elif selccion == 2:
            tipoDeSeleccion="torneo"
            break
        elif selccion == 3:
            tipoDeSeleccion="ranking"
            break
        elif selccion == 4:
            tipoDeSeleccion="aleatorio"
            break
        elif selccion == 5:
            tipoDeSeleccion="sorteo"
            break
        else:
            print("Opción inválida, intente de nuevo.")
            
    #metodo de cruce
    while True:
        print("\nseleccione el metodo de cruce")
        print("1. Segmentado")
        print("2. Multipunto")
        print("3. Uniforme")
        print("4. Basico")
        cruce = int(input("Ingrese el numero de la opcion: "))
        if cruce == 1:
            methodoCruce="Segmentado"
            break
        elif cruce == 2:
            methodoCruce="Multipunto"
            break
        elif cruce == 3:
            methodoCruce="Uniforme"
            break
        elif cruce == 4:
            methodoCruce="Basico"
            break
        else:
            print("Opción inválida, intente de nuevo.")

    #elitismo
    if acMaxMin:
        while True:
            elitismo = input("¿Desea utilizar elitismo? (S/N): ")
            if elitismo == 'S' or elitismo == 's':
                elitismo = True
                break
            elif elitismo == 'N' or elitismo == 'n':
                elitismo = False
                break
            else:
                print("Opción inválida, intente de nuevo.")
    else:
        elitismo = False
    
    ga = GeneticAlgorit(n, Pcruce, Pmuta, generaciones, xRestringe, xObjetivo, infVariables, 
                        elitismo, activarConvergencia, valConvergencia,restriccion,
                        acMaxMin,maximizar,tipoDeSeleccion,methodoCruce)
    
    # capturar todas las salidas de la consola redirigiendolas al archivo de texto
    # Guarda las salidas originales 
    salida_original = sys.stdout
    error_original = sys.stderr

    # Redirige las salidas estándar y de error al archivo
    sys.stdout = archivo
    sys.stderr = archivo
    
    print("\n Funcion Objetivo: \n", xObjetivo)
    print("\n Restricciones: \n", restriccion)
    print("\n Restricciones: \n", xRestringe)
    print("\n informacion Variables de decision: \n", infVariables)    
    print("\n Poblacion: \n", n)
    print("\n Probabilidad de cruce: \n", Pcruce)
    print("\n Probabilidad de mutacion: \n", Pmuta)
    print("\n Generaciones: \n", generaciones)
    print("\n Maximizar o minimizar: \n", acMaxMin)
    if acMaxMin and maximizar:
        print("funcion objetivo a maximizar\n")
    elif acMaxMin and not maximizar:
        print("funcion objetivo a minimizar\n")
    print("\n Elitismo: \n", elitismo)
    print("\n convergencia: \n", activarConvergencia)
    print("\n Porcentaje de convergencia: \n", valConvergencia)
    print("\n\n")
    ga.ejecutar()
    
    # Restaura las salidas originales
    sys.stdout = salida_original
    sys.stderr = error_original

    # Cierra el archivo
    archivo.close()   
    ga.graficar()