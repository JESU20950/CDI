# -*- coding: utf-8 -*-
'''

'''

import numpy as np
import scipy
import imageio
import math 
import scipy.fftpack
import time


import matplotlib.pyplot as plt

from scipy.fftpack import dct, idct



        
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
    (_,N) = p.shape
    c = dctmtx(N)
    #w = c*p*c'
    w = np.dot((np.dot(c,p)),np.transpose(c))
    return w
def idct_bloque_v2(w):
    (_,N) = w.shape
    c = dctmtx(N)
    #p = c'*w*c
    p = np.dot(np.transpose(c),np.dot(w,c))
    return p


'''
Reproducir los bloques base de la transformación para los casos N=4,8
Ver imágenes adjuntas.
'''

def basis_function_blocks(N):
    dct_matrix = dctmtx(N)
    basis = np.zeros((N,N,N,N))
    #obtiene la imagen de los bloques base de la transformación
    for i in range(0,N):
        for j in range(0,N):
            a = np.array([dct_matrix[i,:]]).T
            b = [dct_matrix[j,:]]
            basis[:,:,i,j] =  a*b
    
    #imprime la imagen de los bloques base de la transformación
    k = 1
    for i in range(0,N):
        for j in range (0,N):
            minbase = basis[:,:,i,j].min()
            maxbase = basis[:,:,i,j].max()
            rango = maxbase-minbase
            imagen = ((basis[:,:,i,j]+rango)*(255/(maxbase+rango)))
            plt.subplot(N,N,k)
            plt.imshow(imagen, cmap=plt.cm.gray)
            k = k+1
    plt.show()

basis_function_blocks(2**2)
basis_function_blocks(2**3)

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
    
def dct_division_por_bloque(datos,N):
    (D1,D2) = datos.shape
    for i in range(0,D1,N):
        for j in range(0,D2,N):
            datos_aux = datos[i:(i+N),j:(j+N)]
            datos_aux = dct_bloque_v2(datos_aux)
            datos[i:(i+N),j:(j+N)] = datos_aux
    return datos

def idct_division_por_bloque(datos,N):
    (D1,D2) = datos.shape
    for i in range(0,D1,N):
        for j in range(0,D2,N):
            datos_aux = datos[i:(i+N),j:(j+N)]
            datos_aux = idct_bloque_v2(datos_aux)
            datos[i:(i+N),j:(j+N)] = datos_aux
    return datos


def cuantizacion_division_por_bloque(datos, matrix_cuantitation, N):
    (D1,D2) = datos.shape
    for i in range(0,D1,N):
        for j in range(0,D2,N):
            datos_aux = datos[i:(i+N),j:(j+N)]
            #datous_aux = round(datos_aux./matrix_cuantitation)
            datos_aux = np.round(np.divide(datos_aux,matrix_cuantitation))
            datos[i:(i+N),j:(j+N)] = datos_aux
    return datos

def descuantizacion_division_por_bloque(datos, matrix_cuantitation, N):
    (D1,D2) = datos.shape
    for i in range(0,D1,N):
        for j in range(0,D2,N):
            datos_aux = datos[i:(i+N),j:(j+N)]
            #datous_aux = round(datos_aux./matrix_cuantitation)
            datos_aux = np.multiply(datos_aux,matrix_cuantitation)
            datos[i:(i+N),j:(j+N)] = datos_aux
    return datos


def ratio_compresion_coeficientes_gray(Gray):
    [D1,D2] = Gray.shape
    coeficientes_nulos = np.sum(Gray == 0.)
    coeficientes = D1*D2
    print("ratio compresion", coeficientes/(coeficientes-coeficientes_nulos))
    
def estimacion_error_gray(imagen_gray,imagen_jpeg):
    sigmagray=np.sqrt(sum(sum((imagen_gray-imagen_jpeg)**2)))/np.sqrt(sum(sum((imagen_gray)**2)))
    print("sigmaGray" , sigmagray)
    
    
def jpeg_gris(imagen_gray):
    N = 8
    imagen_jpeg = imagen_gray -128
    imagen_jpeg = dct_division_por_bloque(imagen_jpeg,N)
    imagen_jpeg = cuantizacion_division_por_bloque(imagen_jpeg,Q_Luminance,N)
    ratio_compresion_coeficientes_gray(imagen_jpeg)
    imagen_jpeg = descuantizacion_division_por_bloque(imagen_jpeg,Q_Luminance,N)
    imagen_jpeg = idct_division_por_bloque(imagen_jpeg,N)
    imagen_jpeg = imagen_jpeg +128
    imagen_jpeg = imagen_jpeg.astype(np.uint8)
    estimacion_error_gray(imagen_gray,imagen_jpeg)
    plt.imshow(imagen_jpeg, cmap=plt.cm.gray,vmin=0, vmax=255)
    plt.show()
    return imagen_jpeg

'''
Implementar la función jpeg_color(imagen_color) que: 
1. dibuje el resultado de aplicar la DCT y la cuantización 
(y sus inversas) a la imagen RGB 'imagen_color' 

2. haga una estimación de la ratio de compresión
según los coeficientes nulos de la transformación: 
(#coeficientes/#coeficientes no nulos).

3. haga una estimación del error para cada una de las componentes RGB
Sigma=np.sqrt(sum(sum((imagen_color-imagen_jpeg)**2)))/np.sqrt(sum(sum((imagen_color)**2)))

'''
def fromRGBtoYCbCr(imagen_color):
    R = imagen_color[:,:,0]
    G = imagen_color[:,:,1]
    B = imagen_color[:,:,2]
    Y = 0.299*R+0.587*G+0.114*B
    Cb = -0.1687*R-0.3313*G+0.5*B+128
    Cr = 0.5*R-0.4187*G-0.0813*B+128
    return [Y,Cb,Cr]

def fromYCbCrtoRGB(Y,Cb,Cr):
    R = Y + 1.402*(Cr-128)
    G = Y - 0.71414*(Cr-128) - 0.34414*(Cb-128)
    B = Y+ 1.772*(Cb-128) 
    return [R,G,B]

def ratio_compresion_coeficientes(Y,Cb, Cr):
    [D1,D2] = Y.shape
    coeficientes_nulos = np.sum(Y == 0.) + np.sum(Cb == 0.) + np.sum(Cr == 0.)
    coeficientes = D1*D2*3
    print("ratio compresion", coeficientes/(coeficientes-coeficientes_nulos))
    
def estimacion_error(imagen_color,imagen_jpeg):
    sigmaR =np.sqrt(sum(sum((imagen_color[:,:,0]-imagen_jpeg[:,:,0])**2)))/np.sqrt(sum(sum((imagen_color[:,:,0])**2)))
    sigmaG =np.sqrt(sum(sum((imagen_color[:,:,1]-imagen_jpeg[:,:,1])**2)))/np.sqrt(sum(sum((imagen_color[:,:,1])**2)))
    sigmaB =np.sqrt(sum(sum((imagen_color[:,:,2]-imagen_jpeg[:,:,2])**2)))/np.sqrt(sum(sum((imagen_color[:,:,2])**2)))
    print("sigmaR" , sigmaR)
    print("sigmaG", sigmaG)
    print("sigmaB", sigmaB)
    

def jpeg_color(imagen_color):

    N = 8
    #Convertir de RGB a YCbCr
    [Y,Cb,Cr] = fromRGBtoYCbCr(imagen_color)
    
    #A cada bloque se le aplica la DCT, pero previamente se le resta
    #128 a cada elemento del bloque para que el coeficiente DC se
    #reduzca su valor en 128 · 64 = 8192 en promedio.
    Y = Y-128
    Cb = Cb-128
    Cr = Cr-128

    #DCT por bloques de tamaño N
    Y = dct_division_por_bloque(Y,N)
    Cb = dct_division_por_bloque(Cb,N)
    Cr = dct_division_por_bloque(Cr,N)

    #Se cuantizan los valores obtenidos, con las matrices de
    #cuantización. Los estándares usan una para la componente Y ,
    #luminancia, y otras para las componentes Cb y Cr , crominancias,
    #pero se pueden utilizar otras que se han de incluir.
    Y = cuantizacion_division_por_bloque(Y,Q_Luminance,N)
    Cb = cuantizacion_division_por_bloque(Cb,Q_Chrominance,N)
    Cr = cuantizacion_division_por_bloque(Cr,Q_Chrominance,N)
    
    ratio_compresion_coeficientes(Y,Cb, Cr)
    
    #Descuantizacion
    Y = descuantizacion_division_por_bloque(Y,Q_Luminance,N)
    Cb = descuantizacion_division_por_bloque(Cb,Q_Chrominance,N)
    Cr = descuantizacion_division_por_bloque(Cr,Q_Chrominance,N)
    
    
    #inverse DCT por bloques de tamaño N
    Y = idct_division_por_bloque(Y,N)
    Cb = idct_division_por_bloque(Cb,N)
    Cr = idct_division_por_bloque(Cr,N)

    #Se le suma a cada componente 128
    Y = Y+128
    Cb = Cb+128
    Cr = Cr+128
    
    #Convertir de YCbCr a fromRGB
    [R,G,B] = fromYCbCrtoRGB(Y,Cb,Cr)
    [D1,D2] = R.shape
    imagen_jpeg = np.zeros((D1,D2,3))
    imagen_jpeg[:,:,0] = R
    imagen_jpeg[:,:,1] = G
    imagen_jpeg[:,:,2] = B
    imagen_jpeg = imagen_jpeg.astype(np.uint8)
    estimacion_error(imagen_color,imagen_jpeg)
    plt.imshow(imagen_jpeg)
    plt.show()
    return imagen_jpeg
    
    
   
    
'''
#--------------------------------------------------------------------------
Imagen de GRISES

#--------------------------------------------------------------------------
'''


### .astype es para que lo lea como enteros de 32 bits, si no se
### pone lo lee como entero positivo sin signo de 8 bits uint8 y por ejemplo al 
### restar 128 puede devolver un valor positivo mayor que 128


mandril_gray=imageio.imread('./mandril_gray.png').astype(np.int32)
start= time.clock()
mandril_jpeg=jpeg_gris(mandril_gray)
end= time.clock()
print("tiempo",(end-start))



'''
#--------------------------------------------------------------------------
Imagen COLOR
#--------------------------------------------------------------------------
'''

mandril_color=imageio.imread('./mandril_color.png').astype(np.int32)


start= time.clock()
mandril_jpeg=jpeg_color(mandril_color)     
end= time.clock()
print("tiempo",(end-start))
     
       









