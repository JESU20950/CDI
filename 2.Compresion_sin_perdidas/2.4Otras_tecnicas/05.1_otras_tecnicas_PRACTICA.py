# -*- coding: utf-8 -*-

import math


"""
Dado un mensaje y un alfabeto, dar su codificación usando la técnica 
de 'Move to Front'.


mensaje='mi mama me mima mucho'
alfabeto=[' ', 'a', 'c', 'e', 'h', 'i', 'm', 'o', 'u']
MtFCode(mensaje,alfabeto)=[6, 6, 2, 2, 3, 1, 1, 2, 2, 5, 2, 
                           2, 4, 1, 4, 3, 2, 8, 6, 7, 8]

"""

def MtFCode(mensaje,alfabeto):
    code = []
    for c in mensaje:
        index = alfabeto.index(c)
        del alfabeto[index]
        alfabeto = [c] + alfabeto
        code = code + [index]
    return code


mensaje='mi mama me mima mucho'
alfabeto=[' ', 'a', 'c', 'e', 'h', 'i', 'm', 'o', 'u']
print(MtFCode(mensaje,alfabeto))



"""
Dado un mensaje, aplicar la transformación de Burrows-Wheeler
devolviendo la última columna y la posición.

mensaje='cadacamacasapasa'
BWT(mensaje)=('sdmccspcaaaaaaaa', 8)

"""


def get_rotation_sorted(mensaje):
    rotation = []
    for i in range(0, len(mensaje)):
        rotation = rotation + [mensaje]
        mensaje = mensaje[1:]+mensaje[0]
    rotation.sort()
    return rotation
    
def BWT(mensaje):
    rotations_sorted = get_rotation_sorted(mensaje)
    posicion = rotations_sorted.index(mensaje)
    ultima_columna = [  rotation[len(mensaje)-1] for rotation in rotations_sorted]
    ultima_columna = "".join(ultima_columna)
    return ultima_columna, posicion    


mensaje='cadacamacasapasa'
print(BWT(mensaje))



"""
1. Hallar la tabla de frecuencias asociada a los símbolos del 
fichero que contiene el texto de 'La Regenta' y calcular la 
entropía. 

mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte.'
tabla_frecuencias=[[' ', 23], [',', 2], ['.', 2], ['E', 1], ['L', 1], ['N', 1], ['S', 1], ['a', 18], ['b', 4], ['c', 6], ['d', 3], ['e', 15], ['g', 1], ['h', 2], ['i', 7], ['j', 1], ['l', 7], ['m', 2], ['n', 6], ['o', 7], ['p', 2], ['q', 2], ['r', 9], ['s', 8], ['t', 4], ['u', 6], ['v', 1], ['y', 1], ['z', 1], ['í', 1]]
Entropia = 4.224930298009863
"""

'''
Dada una lista de frecuencias n, halla su entropía.
'''
def H2(n):
    suma_frecuencias = sum(n)
    suma = 0
    for f in n:
        if (f > 0):
            p = (f/suma_frecuencias)
            suma = suma + p*math.log(p,2)
    return -suma
def tablaFrecuencias(mensaje,numero_de_simbolos=1):
    frecuencias = dict([])
    for i in range(0,len(mensaje),numero_de_simbolos):
        if (i+numero_de_simbolos-1<len(mensaje)):
            simbolo = mensaje[i:(i+numero_de_simbolos)] 
            if simbolo in frecuencias:
                frecuencias[simbolo] = frecuencias[simbolo] +1
            else:
                frecuencias[simbolo] = 1
        else:
                frecuencias[mensaje[i:len(mensaje)]] = 1
    lista = list(map(list, frecuencias.items()))
    return lista

def ejercicio1():
    mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte.'
    tablafrecuencias = tablaFrecuencias(mensaje,1)
    frecuencias = [f for [_,f] in tablafrecuencias]
    print('Ejercicio1\n','Entropia mensaje pequeña parte:', H2(frecuencias)) 
    with open ("la_regenta.txt", "r") as myfile:
        mensaje=myfile.read()
    tablafrecuencias = tablaFrecuencias(mensaje,1)
    frecuencias = [f for [_,f] in tablafrecuencias]
    print('Ejercicio1\n','Entropia mensaje completo:',H2(frecuencias)) 

ejercicio1()


"""
2. A continuación aplicar la transformación de Burrows-Wheeler 
al fichero (primero probad con una parte antes de hacerlo 
con el fichero entero) y a continuación aplicar MtFCode a 
la última columna obtenida. Hallar la tabla de frecuencias 
del resultado obtenido y calcular la entropía.


mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte.'
BWT(mensaje)='.lons,alda,raaasyseealeroae .  ciLíbltjghd cbllnraau i ae  au suttu  iirphbirs  colvsccueaE b aeraiaee tsrdcNz m neu reoeooeaa a oesnrnniqqpS  em'
alfabeto=[' ', ',', '.', 'E', 'L', 'N', 'S', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'y', 'z', 'í']
MtFCode(BWT(mensaje),alfabeto)=[2, 16, 19, 19, 23, 6, 11, 5, 14, 2, 3, 23, 2, 0, 0, 5, 27, 1, 17, 0, 3, 7, 2, 5, 9, 4, 3, 11, 11, 1, 0, 17, 20, 15, 29, 19, 11, 26, 23, 22, 23, 19, 11, 11, 8, 8, 0, 20, 17, 16, 0, 27, 7, 15, 1, 3, 17, 2, 0, 2, 4, 2, 19, 2, 15, 0, 1, 3, 0, 6, 0, 7, 26, 14, 12, 4, 4, 8, 6, 0, 13, 20, 14, 28, 5, 4, 0, 11, 14, 14, 24, 9, 12, 1, 3, 4, 11, 2, 12, 1, 3, 0, 4, 15, 10, 6, 17, 11, 25, 29, 7, 28, 1, 20, 10, 15, 3, 9, 3, 18, 1, 1, 0, 1, 13, 0, 4, 1, 1, 3, 3, 12, 7, 6, 1, 0, 14, 29, 0, 21, 29, 9, 0, 8, 12]

Nuevo_alfabeto=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
Nueva_tabla_frecuencias=[[0, 18], [1, 13], [2, 9], [3, 11], [4, 8], [5, 4], [6, 5], [7, 5], [8, 4], [9, 4], [10, 2], [11, 9], [12, 5], [13, 2], [14, 6], [15, 5], [16, 2], [17, 5], [18, 1], [19, 5], [20, 4], [21, 1], [22, 1], [23, 4], [24, 1], [25, 1], [26, 2], [27, 2], [28, 2], [29, 4]]
Nueva entropia = 4.507878869023793
"""
def get_alphabet_from_message(message):
    alphabet = set({})
    for c in message:
        alphabet.add(c)
    return list(alphabet)

def tablaFrecuencias_lista(codigo):
    frecuencias = dict([])
    for simbolo in codigo:       
        if simbolo in frecuencias:
            frecuencias[simbolo] = frecuencias[simbolo] +1
        else:
            frecuencias[simbolo] = 1
    lista = list(map(list, frecuencias.items()))
    return lista  


def ejercicio2():
    mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte.'
    [nuevo_mensaje,_] = BWT(mensaje)
    alfabeto = get_alphabet_from_message(nuevo_mensaje)
    alfabeto.sort()
    nuevo_mensaje = MtFCode(nuevo_mensaje,alfabeto)
    nueva_tabla_frecuencias = tablaFrecuencias_lista(nuevo_mensaje)
    nuevas_frecuencias = [f for [_,f] in nueva_tabla_frecuencias]
    print('Ejercicio2\n','Entropia mensaje pequeña parte:', H2(nuevas_frecuencias)) 
ejercicio2()



"""
Observar qué pasa a medida que el mensaje se acerca al texto entero.
"""
