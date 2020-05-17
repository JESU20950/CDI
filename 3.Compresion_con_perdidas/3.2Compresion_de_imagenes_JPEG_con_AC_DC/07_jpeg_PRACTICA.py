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

#basis_function_blocks(2**3)
    

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

def resize_mod_rgb(imagen_color,N):
    (D1,D2,_) = imagen_color.shape
    if (D1%N != 0):
        D1_addition = N-(D1%N)
        last = image_color[(D1-1),:,:]
        last = np.repeat(last,repeats = D1_addition-1, axis = 0)
        imagen_color = np.concatenate((imagen_color,last),axis = 0)
    if (D2%N != 0):
        D2_Addition = N-(D2%N)
        last = image_color[:,(D2-1),:]        
        last = np.repeat(last,repeats = D2_addition-1, axis = 1)
        imagen_color = np.concatenate((imagen_color,last),axis = 1)
    return imagen_color
        
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
            datos_aux = np.around(np.divide(datos_aux,matrix_cuantitation))
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

def binari(number, digits):
    binari_number = "{0:b}".format(number)
    rest = digits-len(binari_number)
    for i in range(0, rest):
        binari_number = '0'+binari_number
    
    return binari_number

def binary_DC_code(number):
    if(number == 2**15):
        return ['1111111111111111','0000000000000000']
    if(number == 0):
        return '0'

    row = math.floor(math.log(abs(number),2)+1)    
    coderow = '0'
    for i in range(0,row):
        coderow = '1'+coderow
        
    column = number
    if (number <0):
        column = ((2**row)-1)+number
        
    codecolumn =  binari(column,row)
    return [coderow,codecolumn]
        
    
def codificacion_DC(data,N):
    (D1,D2) = data.shape
    DCs = np.array((D1,D2))
    for i in range(0,D1,N):
        for j in range(0,D2,N):
            if(i == j and i == 0):
                [coderow,codecolumn] = binary_DC_code(data[i,j])
                DCs[i,j] = coderow+codecolumn
                anterior = data[i,j]
            else:
                diferencia = anterior-data[i,j] 
                [coderow,codecolumn] = binary_DC_code(diferencia)
                DCs[i,j] = coderow+codecolumn
                anterior = data[i,j]
    return DCs
def codificacion_AC(data,N):
    (D1,D2) = data.shape
    ACs = np.array(N*N)
    
    for i in range(0,D1,N):
        for j in range(0,D2,N):
            resultado = ''
            z = 0
            for ii in range(i,i+N):
                if (ii == 0):
                    for jj in range(j+1,j+N):
                        jj_aux = jj
                        ii_aux = ii
                        while(jj_aux >= 0):
                            if (data[ii_aux,jj_aux] == 0):
                                z = z+1
                            else:
                                resultado = resultado + binary_AC_code(data[ii_aux,jj_aux],z)
                                z = 0
                            jj_aux = jj_aux-1
                            ii_aux = jj_aux-1
                else:
                    jj_aux = j+N
                    ii_aux = ii
                    while(jj_aux >= 0):
                        if (data[ii_aux,jj_aux] == 0):
                                z = z+1
                        else:
                            resultado = resultado + binary_AC_code(data[ii_aux,jj_aux],z)
                            z = 0
                        jj_aux = jj_aux-1
                        ii_aux = ii_aux-1
            
            
def codificacion_DC_AC(data,N):
    DCs = codificacion_DC(data,N)
    ACs = codificacion_AC(data,N)
    
            
def jpeg_color_compression(imagen_color,N=8):
    #Si el numero de filas o columnas no es multiplo de N, se replica la última fila/columnas
    (D1,D2,_) = imagen_color.shape
    imagen_color = resize_mod_rgb(imagen_color,N)
    
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
    
    #Se tratan los coeficientes DC y AC por separado. Todos los DC se
    #codifican juntos (sus diferencias); los 63 AC se codifican juntos
    Y = codificacion_DC_AC(Y,N)
    Cb = codificacion_DC_AC(Cb,N)
    Cr = codificacion_DC_AC(Cb,N)
    
    
    
    
    
    
    
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
    imagen_color = np.zeros((D1,D2,3))
    imagen_color[:,:,0] = R
    imagen_color[:,:,1] = G
    imagen_color[:,:,2] = B
    imagen_color = imagen_color.astype(np.uint8)
    plt.imshow(imagen_color)
    plt.show()
    return


import sys
np.set_printoptions(threshold=sys.maxsize)
'''
#--------------------------------------------------------------------------
Imagen de GRISES

#--------------------------------------------------------------------------

'''


### .astype es para que lo lea como enteros de 32 bits, si no se
### pone lo lee como entero positivo sin signo de 8 bits uint8 y por ejemplo al 
### restar 128 puede devolver un valor positivo mayor que 128
'''
mandril_gray=scipy.ndimage.imread('mandril_gray.png').astype(np.int32)

start= time.clock()
mandril_jpeg=jpeg_gris(mandril_gray)
end= time.clock()
print("tiempo",(end-start))
'''


'''
#--------------------------------------------------------------------------
Imagen COLOR
#--------------------------------------------------------------------------
'''

mandril_color=imageio.imread('./mandril_color.png').astype(np.int32)


print(binary_DC_code((2**15)-1))
print(binary_DC_code(-4))
print(binary_DC_code(4))
start= time.clock()
#mandril_jpeg=jpeg_color_compression(mandril_color,8)     
end= time.clock()
print("tiempo",(end-start))
     
       









