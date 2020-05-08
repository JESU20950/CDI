# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:14:40 2020

@author: fernando
"""

########################################################
from scipy import misc
import scipy
import numpy as np
import time
import matplotlib.pyplot as plt
import imageio
import PIL
import pickle



#%%


imagen=imageio.imread('../standard_test_images/jetplane.png')
# imagen=imageio.imread('../standard_test_images/mandril_gray.png')
# imagen=imageio.imread('../standard_test_images/crosses.png')
# imagen=imageio.imread('../standard_test_images/circles.png')
# imagen=imageio.imread('../standard_test_images/cameraman.png')
# imagen=imageio.imread('../standard_test_images/walkbridge.png')


(n,m)=imagen.shape # filas y columnas de la imagen

plt.figure()
plt.xticks([])
plt.yticks([])
plt.imshow(imagen, cmap=plt.cm.gray,vmin=0, vmax=255)
#plt.show()

       

#%%
"""
Definir una funcion que dada una imagen
cuantize uniformemente los valores en cada bloque 
"""


def cuantizacion_bloque(bloque, bits):
    minimo = min(map(min, bloque))
    maximo = max(map(max,bloque))
    distance = (maximo-minimo)/(2**bits)
    intervalo = [minimo]
    for _ in range(0, (2**bits)-1):
        intervalo = intervalo + [intervalo[len(intervalo)-1]+distance]
    intervalo = intervalo + [maximo+1]
    (n,m) = bloque.shape
    bloqueCodificado = [] 
    for i in range(0, n):
        fila = []
        for j in range(0, m):
            number = bloque[i,j]
            for c in range(0,len(intervalo)-1):
                if (number>= intervalo[c] and number<intervalo[c+1]):
                    fila = fila + [c]
        bloqueCodificado = bloqueCodificado + [fila]
    return [[[minimo, maximo ] , np.array(bloqueCodificado) ]]
    

def Cuantizacion_uniforme_adaptativa(imagen, bits=3, n_bloque=8):
    (n,m)=imagen.shape
    imagenCodigo = [[n,m,n_bloque,bits]]
    for i in range(0,n, n_bloque):
        for j in range(0,m, n_bloque):
            bloque = imagen[i:(i+n_bloque), j:(j+n_bloque)]
            cuantizacion = cuantizacion_bloque(bloque, bits)
            imagenCodigo = imagenCodigo + cuantizacion
    return imagenCodigo


"""
imagen: imagen a cuantizar
bits: número de bits necesarios para cuantizar cada bloque, 
      o sea que en cada bloque habrá 2**bits valores diferentes como máximo
n_bloque: se consideran bloques de tamaño n_bloque*n_bloque 

imagenCodigo: es una lista de la forma
[[n,m,n_bloque,bits],[[minimo,maximo],bloqueCodificado],...,[[minimo,maximo],bloqueCodificado]]

siendo:
[n,m,n_bloque,bits] información de la imagen
  n: número de filas de la imagen
  m: número de columnas de la imagen
  n_bloque: tamaño de los bloques usados (múltiplo de n y m)
Ejemplo: [1024, 1024, 8, 3]

[[minimo,maximo],bloqueCodificado] información de cada bloque codificado
minimo: valor mínimo del bloque
maximo: valor máximo del bloque
bloqueCodificado: array de tamaño n_bloque*n_bloque que contiene en cada 
  posición a que intervalo de cuantización correspondía el valor del píxel
  correspondiente en la imagen

Ejemplo: sabemos que trabajamos con bloques 8x8 y que hemos cuantizado en 2**3=8 niveles
    [[85, 150], 
    Array([[4, 0, 0, 4, 7, 7, 6, 7], 
           [4, 3, 1, 1, 4, 7, 7, 6],
           [6, 6, 3, 0, 0, 4, 6, 6], 
           [6, 6, 5, 3, 1, 0, 3, 6],
           [6, 5, 6, 6, 4, 0, 0, 3],
           [5, 6, 6, 6, 6, 4, 2, 0],
           [6, 6, 5, 5, 6, 7, 4, 1],
           [6, 6, 5, 5, 5, 6, 6, 5]]  
   El valor mínimo de los píxeles del bloque era 85 y el máximo 150, por lo 
   tanto los límites de decisión son:
       [85.0, 93.25, 101.5, 109.75, 118.0, 126.25, 134.5, 142.75, 151.0]
   el valor del primer pixel (4) estaría entre 109.75<=p<118
   el valor del segundo pixel pixel (0) estaría entre 85<=p<93.25...
   
      
Importante: Trabajar con Arrays de Numpy

"""


#%%
"""
Definir una funcion que dada una imagen codificada por la función 
Cuantizacion_uniforme_adaptativa() muestre la imagen

"""
def decuantizacion_bloque(bloque, bits):
    [[minimo,maximo],bloqueCodificado] = bloque
    distance = (maximo-minimo)/(2**bits)
    f = lambda number: minimo+(distance*number)+distance/2
    bloquecuantizacion = [f(bloqueCodificado[0])]
    for numbers in bloqueCodificado[1:]:
        fila = f(numbers)
        bloquecuantizacion = np.concatenate((bloquecuantizacion, [fila]), axis=0)
    return bloquecuantizacion

def Dibuja_imagen_cuantizada(imagenCodigo):
    [n,m,n_bloque,bits] =  imagenCodigo[0]
    imagenCodigo = imagenCodigo[2:]
    imagen = [decuantizacion_bloque(imagenCodigo[1], bits)]
    for bloque in imagenCodigo:
        c = decuantizacion_bloque(bloque, bits)
        (_,m_aux) = imagen[len(imagen)-1].shape
        if (m_aux == m):
            imagen = imagen + [c]
        else:
            imagen[len(imagen)-1] = np.concatenate((imagen[len(imagen)-1],c), axis= 1)
    imagen_final = imagen[0]
    for fragmento_imagen in imagen[1:]:
        imagen_final = np.concatenate((imagen_final,fragmento_imagen), axis = 0)
    return imagen_final
def ratio_compresion(imagen,bits,n_bloque):
    (n,m)=imagen.shape 
    size_imagen = n*m*8
    size_imagenCodigo = (n*m*bits)+ 2*32*(m/n_bloque)
    return (size_imagen/size_imagenCodigo)
 #%%   
"""
Aplicar vuestras funciones a las imágenes que encontraréis en la carpeta 
standard_test_images hacer una estimación de la ratio de compresión
"""
bits=2
n_bloque = 8
#q=2**(bits) 
#imagen2=((np.floor(imagen/q)+1/2).astype(np.uint8))*q

imagenCodigo = Cuantizacion_uniforme_adaptativa(imagen, bits, n_bloque)
with open('out.txt', 'w') as f:
	print(imagenCodigo,file=f)
imagen2 = Dibuja_imagen_cuantizada(imagenCodigo)
imagenPIL=PIL.Image.fromarray(imagen2)
imagenPIL.convert('RGB').save('resultado.png')
print(ratio_compresion(imagen,bits,n_bloque))

####################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#%%
"""
Algunas sugerencias que os pueden ser útiles
"""

# Divido todos los píxeles de la imagen por q
# a continuación redondeo todos los píxeles
# a continuación sumo 1/2 a todos los píxeles de la imagen
# a continuación convierto los valores de todos los píxeles en enteros de 8 bits sin signo
# por último múltiplico todos los píxeles de la imagen por q
"""
bits=3
q=2**(bits) 
imagen2=((np.floor(imagen/q)+1/2).astype(np.uint8))*q

# dibujo la imagen cuanzizada resultante

fig=plt.figure()
fig.suptitle('Bloques: '+str(bits)+' bits/píxel')
plt.xticks([])
plt.yticks([])
plt.imshow(imagen2, cmap=plt.cm.gray,vmin=0, vmax=255) 
#plt.show()


# Lectura y escritura de objetos

import pickle

fichero='QScalar'

with  open(fichero+'_dump.pickle', 'wb') as file:
    pickle.dump(imagenCodigo, file)


with open(fichero, 'rb') as file:
    imagenLeidaCodificada=pickle.load(file)


# Convertir un array en imagen, mostrarla y guardarla en formato png.
# La calidad por defecto con que el ecosistema python (ipython, jupyter,...)
# muestra las imágenes no hace justicia ni a las imágenes originales ni a 
# las obtenidas tras la cuantización.     

import PIL

imagenPIL=PIL.Image.fromarray(imagen2)
imagenPIL.show()
imagenPIL.save(fichero +'_imagen.png', 'PNG')
"""
