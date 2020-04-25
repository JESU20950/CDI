# -*- coding: utf-8 -*-
"""

"""
import math
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.cluster.vq import vq, kmeans
import imageio

#%%
lena=imageio.imread('../standard_test_images/lena_gray_512.png')
(n,m)=lena.shape # filas y columnas de la imagen
plt.figure()    
plt.imshow(lena, cmap=plt.cm.gray)
#plt.show()



#%%
   
"""
Usando K-means http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
crear un diccionario cuyas palabras sean bloques 8x8 con 512 entradas 
para la imagen de Lena.

Dibujar el resultado de codificar Lena con dicho diccionario.

Calcular el error, la ratio de compresión y el número de bits por píxel
"""

def Cuantizacion_vectorial(imagen, n_bloque=8, n_centroids=2**9):
    (n,m)=imagen.shape
    points = []
    for i in range(0,n, n_bloque):
        for j in range(0,m, n_bloque):
            bloque = imagen[i:(i+n_bloque), j:(j+n_bloque)]
            point = []
            for x in range(0, n_bloque):
                for y in range(0, n_bloque):
                    point = point + [float(bloque[x,y])]
            points = points + [point]
    points = np.array(points)
    [codebook, distortion] = kmeans(points,n_centroids)
    [code, dist] = vq(points, codebook)
    return [code,codebook]

def Dibuja_imagen_cuantizada(imagenCodigo, m, n_bloque):
    [code,codebook] = imagenCodigo
    subimagen = np.array(codebook[code[0]])
    subimagen = np.reshape(subimagen, (n_bloque,n_bloque))
    imagen = [subimagen]
    for c in code[1:]:
        subimagen = np.array(codebook[c])
        subimagen = np.reshape(subimagen, (n_bloque,n_bloque))
        (_,m_aux) = imagen[len(imagen)-1].shape
        if (m_aux == m):
            imagen = imagen + [subimagen]
        else:
            imagen[len(imagen)-1] = np.concatenate((imagen[len(imagen)-1], subimagen), axis= 1)
    imagen_final = imagen[0]
    for fragmento_imagen in imagen[1:]:
        imagen_final = np.concatenate((imagen_final,fragmento_imagen), axis = 0)
    return imagen_final



def calculos_imagenes(imagen_original,imagenCodigo, n_bloque, n_centroids):
    [code,codebook] = imagenCodigo
    length = code.size
    (n,m)=imagen_original.shape
    imagen = Dibuja_imagen_cuantizada(imagenCodigo, m, n_bloque) 
    error = sum(sum(abs(imagen_original-imagen)))
    
    size_original_image = n*m*8
    n_bits_centroids = math.ceil(math.log(n_centroids,2))
    size_code = (code.size*n_bits_centroids) + (n_centroids*8*n_bloque*n_bloque)
    size_original_image = 8*n*m
    ratio_compresion = size_original_image/size_code
    bits_por_pixel = size_code/(n*m)
    return [error, ratio_compresion, bits_por_pixel ] 



n_bloque = 8
n_centroids = 2**9
    
imagenCodigo = Cuantizacion_vectorial(lena, n_bloque, n_centroids)
imagen = Dibuja_imagen_cuantizada(imagenCodigo, m, n_bloque)
plt.imshow(imagen, cmap=plt.cm.gray)
plt.show()

[error, ratio_compresion, bits_por_pixel ]  = calculos_imagenes(lena,imagenCodigo, n_bloque, n_centroids)

print('error=', error)
print('ratio_compresion=' , ratio_compresion)
print('bits_por_pixel=' , bits_por_pixel)

"""
Hacer lo mismo con la imagen Peppers (escala de grises)

http://atenea.upc.edu/mod/folder/view.php?id=1577653
http://www.imageprocessingplace.com/downloads_V3/root_downloads/image_databases/standard_test_images.zip
"""

n_bloque = 8
n_centroids = 2**9
pepper = imageio.imread('../standard_test_images/peppers_gray.png')
(n,m)=lena.shape # filas y columnas de la imagen
imagenCodigo = Cuantizacion_vectorial(pepper, n_bloque, n_centroids)
imagen = Dibuja_imagen_cuantizada(imagenCodigo, m, n_bloque)
plt.imshow(imagen, cmap=plt.cm.gray)
plt.show()

[error, ratio_compresion, bits_por_pixel ]  = calculos_imagenes(pepper,imagenCodigo, n_bloque, n_centroids)

print('error=', error)
print('ratio_compresion=' , ratio_compresion)
print('bits_por_pixel=' , bits_por_pixel)


"""
Dibujar el resultado de codificar Peppers con el diccionarios obtenido
con la imagen de Lena.

Calcular el error.
"""
n_bloque = 8
n_centroids = 2**9
lena=imageio.imread('../standard_test_images/lena_gray_512.png')
pepper = imageio.imread('../standard_test_images/peppers_gray.png')
imagenCodigo = Cuantizacion_vectorial(lena, n_bloque, n_centroids)
imagenCodigo2 = Cuantizacion_vectorial(pepper, n_bloque, n_centroids)
imagenCodigofinal = [imagenCodigo[0],imagenCodigo2[1]]
imagen = Dibuja_imagen_cuantizada(imagenCodigofinal, m, n_bloque)
plt.imshow(imagen, cmap=plt.cm.gray)
plt.show()

[error, ratio_compresion, bits_por_pixel ]  = calculos_imagenes(pepper,imagenCodigofinal, n_bloque, n_centroids)

print('error=', error)
print('ratio_compresion=' , ratio_compresion)
print('bits_por_pixel=' , bits_por_pixel)






