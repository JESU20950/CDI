# -*- coding: utf-8 -*-
'''

'''

import numpy as np
import scipy
import imageio
import math 
import scipy.fftpack



import matplotlib.pyplot as plt




        
'''
Matrices de cuantización, estándares y otras
'''

    
Q_Luminance=np.array([
[16 ,11, 10, 16,  24,  40,  51,  61],
[12, 12, 14, 19,  26,  58,  60,  55],
[14, 13, 16, 24,  40,  57,  69,  56],
[14, 17, 22, 29,  51,  87,  80,  62],
[18, 22, 37, 56,  68, 109, 103,  77],
[24, 35, 55, 64,  81, 104, 113,  92],
[49, 64, 78, 87, 103, 121, 120, 101],
[72, 92, 95, 98, 112, 100, 103, 99]])

Q_Chrominance=np.array([
[17, 18, 24, 47, 99, 99, 99, 99],
[18, 21, 26, 66, 99, 99, 99, 99],
[24, 26, 56, 99, 99, 99, 99, 99],
[47, 66, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99],
[99, 99, 99, 99, 99, 99, 99, 99]])

def Q_matrix(r=1):
    m=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            m[i,j]=(1+i+j)*r
    return m

'''
Implementar la DCT (Discrete Cosine Transform) 
y su inversa para bloques NxN (podéis utilizar si quereis el paquete scipy.fftpack)

dct_bloque(p,N)
idct_bloque(p,N)

p bloque NxN

'''
def dct_bloque(p):
    (n,_) = p.shape
    w = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            ci = 1
            cj = 1
            if (i == 0):
                ci = 1/math.sqrt(2)
            if (j == 0):
                cj = 1/math.sqrt(2)
            suma = 0
            for x in range(0,n):
                for y in range(0,n):
                    suma = suma + (p[x,y]* math.cos( (2*x+1)*i*math.pi/(2*n))* math.cos( (2*y+1)*j*math.pi/(2*n)))
            
            w[i,j] = (1/math.sqrt(2*n))*ci*cj*suma
    return w

def idct_bloque(w):
    (n,_) = w.shape
    p = np.zeros((n,n))
    for x in range(0,n):
        for y in range(0,n):
            suma = 0
            for i in range(0,n):
                for j in range(0,n):
                    ci = 1
                    cj = 1
                    if (i == 0):
                        ci = 1/math.sqrt(2)
                    if (j == 0):
                        cj = 1/math.sqrt(2)
                    suma = suma + (ci*cj*w[i,j]* math.cos( (2*x+1)*i*math.pi/(2*n))* math.cos( (2*y+1)*j*math.pi/(2*n)))
            p[x,y] = (1/math.sqrt(2*n))*suma
    return p

def dctmtx(N):
    dct_matrix = np.zeros((N,N));
    for i in range(0,N):
        for j in range(0,N):
            if (i == 0):
                dct_matrix[i,j] = 1/math.sqrt(N)
            else:
                dct_matrix[i,j] = math.sqrt(2/N)*math.cos((2*j+1)*i*math.pi/(2*N))
    return dct_matrix

def dct_bloque_v2(p):
    (N,_) = p.shape
    c = dctmtx(N)
    w = c*p*np.transpose(c)
    return w

def idct_bloque_v2(w):
    (N,_) = p.shape
    c = dctmtx(N)
    p = np.transpose(c)*w*c
    return p

'''
Reproducir los bloques base de la transformación para los casos N=4,8
Ver imágenes adjuntas.
'''

            
    
    
def basis_function_blocks(N):
    dct_matrix = dctmtx(N)
    zeros = np.zeros((N,N,N,N))
    for i in range(0,N)
        for 
    
    
    
#basis_function_images(2**3)
    

'''
Implementar la función jpeg_gris(imagen_gray) que: 
1. dibuje el resultado de aplicar la DCT y la cuantización 
(y sus inversas) a la imagen de grises 'imagen_gray' 

2. haga una estimación de la ratio de compresión
según los coeficientes nulos de la transformación: 
(#coeficientes/#coeficientes no nulos).

3. haga una estimación del error
Sigma=np.sqrt(sum(sum((imagen_gray-imagen_jpeg)**2)))/np.sqrt(sum(sum((imagen_gray)**2)))


'''

def jpeg_gris(imagen_gray):
    return

''''
Implementar la función jpeg_color(imagen_color) que: 
1. dibuje el resultado de aplicar la DCT y la cuantización 
(y sus inversas) a la imagen RGB 'imagen_color' 

2. haga una estimación de la ratio de compresión
según los coeficientes nulos de la transformación: 
(#coeficientes/#coeficientes no nulos).

3. haga una estimación del error para cada una de las componentes RGB
Sigma=np.sqrt(sum(sum((imagen_color-imagen_jpeg)**2)))/np.sqrt(sum(sum((imagen_color)**2)))

'''

def jpeg_color(imagen_color):
    return

'''
#--------------------------------------------------------------------------
Imagen de GRISES

#--------------------------------------------------------------------------
'''


### .astype es para que lo lea como enteros de 32 bits, si no se
### pone lo lee como entero positivo sin signo de 8 bits uint8 y por ejemplo al 
### restar 128 puede devolver un valor positivo mayor que 128

mandril_gray=scipy.ndimage.imread('mandril_gray.png').astype(np.int32)

start= time.clock()
mandril_jpeg=jpeg_gris(mandril_gray)
end= time.clock()
print("tiempo",(end-start))


'''
#--------------------------------------------------------------------------
Imagen COLOR
#--------------------------------------------------------------------------
'''

mandril_color=scipy.misc.imread('./mandril_color.png').astype(np.int32)



start= time.clock()
mandril_jpeg=jpeg_color(mandril_color)     
end= time.clock()
print("tiempo",(end-start))
     
       









