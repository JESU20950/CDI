# -*- coding: utf-8 -*-
"""

"""
import math
import numpy as np
import matplotlib.pyplot as plt


'''
Dada una lista p, decidir si es una distribución de probabilidad (ddp)
0<=p[i]<=1, sum(p[i])=1.
'''
def es_ddp(p,tolerancia=10**(-5)):
    suma = 0
    for i in p:
        if (i < 0 or i > 1):
            return False
        suma = suma + i
    return (suma< 1+tolerancia and suma > 1-tolerancia) 

    



'''
Dado un código C y una ddp p, hallar la longitud media del código.
'''

def LongitudMedia(C,p):
    suma = 0
    for i in range(0,len(C)):
        suma = suma + len(C[i])*p[i]
    return suma

        
    
'''
Dada una ddp p, hallar su entropía.
'''
def H1(n):
    suma = 0
    for p in n:
        if (p > 0):
            suma = suma+ (math.log(p,2))*p
    return -suma
        

'''
Dada una lista de frecuencias n, hallar su entropía.
'''
def H2(n):
    print(n)
    suma_frecuencias = sum(n)
    suma = 0
    for f in n:
        if (f > 0):
            p = (float(f)/float(suma_frecuencias))
            suma = suma + p*math.log(p,2)
    return -suma




'''
Ejemplos
'''
C=['001','101','11','0001','000000001','0001','0000000000']
p=[0.5,0.1,0.1,0.1,0.1,0.1,0]
n=[5,2,1,1,1]

#print(es_ddp(p))
#print(H1(p))
#print(H2(n))
#print(LongitudMedia(C,p))



'''
Dibujar H([p,1-p])
'''
x = []
y = []
for p in range(0,100,1):
    p2 = p/100
    x.append(p2)
    y.append(H1([p2,1-p2]))
plt.plot(x,y)
plt.show() 






