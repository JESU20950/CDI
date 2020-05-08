from scipy import misc
import scipy
import numpy as np
import time
import matplotlib.pyplot as plt
import imageio
import PIL
import pickle

imagen1=imageio.imread('../3.1.1Cuantizacion escalar/QScalar_jetplane_2bits.png')
imagen2=imageio.imread('../3.1.1Cuantizacion escalar/resultado.png')

(n,m,_)=imagen2.shape

for i in range(0,n):
    for j in range(0,m):
        print(imagen1[i,j] != imagen2[i,j])
        if ((imagen1[i,j] != imagen2[i,j]).any):
            print("diferentes")
            break
