# -*- coding: utf-8 -*-
"""

@author: martinez
"""

import termcolor



"""
Función auxiliar para debugar
"""
# def escribe(entrada,debug):
#     if debug: print(entrada)
"""
Funciones auxiliares para pintar texto en color
"""

def blue(mensaje):
    return termcolor.colored(mensaje,'blue')
def green(mensaje):
    return termcolor.colored(mensaje,'green')
def red(mensaje):
    return termcolor.colored(mensaje,'red')


"""
Funciones para debugar
"""   
def escribe_contextos_antes(mensaje, j, contextos,orden,debug=False):
    if debug:
        print("\n+++++++++++++++++++++++++++")
        print(green(mensaje[:max(j-orden,0)])+blue(mensaje[max(j-orden,0):j]),red(mensaje[j]),mensaje[min(j+1,len(mensaje)):])
        for k in range(orden+1):
           for i in contextos: 
               if len(i)==k: 
                   if i!='':                   
                       print(i,contextos[i])
                   else: #para no escribir el '':0 que he tenido que añadir al inicializar el contexto
                      asd={}
                      for ii in contextos[i]:
                          if ii!='': asd[ii]=contextos[i][ii]
                      print(i,asd)      
                      
def escribe_contextos_despues(contextos,orden,debug=False):
    if debug:
        print("Contexto actualizado:")
        for k in range(orden+1):
           for i in contextos: 
               if len(i)==k: 
                   if i!='':                   
                       print(i,contextos[i])
                   else: #para no escribir el '':0 que he tenido que añadir al inicializar el contexto
                      asd={}
                      for ii in contextos[i]:
                          if ii!='': asd[ii]=contextos[i][ii]
                      print(i,asd)        
    
    
    
"""
Dado un mensaje llama a PPM para generar una lista de items
para pasar a codificación aritmética/huffman, los item son de la forma:
[[tamaño del contexto,letra a codificar],[alfabeto],[frecuencias]] ó
[[$,letra a codificar],[],[]] si es la primera vez que aparece la letra.
"""    
    
def PPMCode(mensaje,orden=2,debug=False):
    contextos={'':{'':0}} #guardo contexto y contador  
    codigo=[]
    n=len(mensaje)
    for i in range(n):     
            contextoActual=mensaje[max(0,i-orden):i]
            letra=mensaje[i]
            escribe_contextos_antes(mensaje, i, contextos,orden,debug)
            contextos, tamanyo_del_contexto_y_letra, alfabeto,probabilidades=PPM(letra,contextoActual,contextos,debug)
            codigo.append([tamanyo_del_contexto_y_letra, alfabeto,probabilidades])
            escribe_contextos_despues(contextos,orden,debug)
    return contextos, codigo


"""
Dada una letra (la siguiente en el mensaje), el contextoActual (k letras anteriores) 
y la lista de contextos con sus frecuencias, devuelve la lista actualizada de 
contextos con sus nuevas frecuencias, el tamaño del contexto y 
la letra nueva--aunque ya la sabíamos--,, el alfabeto correspondiente al contexto
particular en el que se codificará la letra con sus frecuencias (si es la
primera aparición de la letra devuelve $ en vez del tamaño del contexto 
y tanto el alfabeto como las frecuencias están vacías)

"""

def PPM(letra,contextoActual,contextos,debug=False):
    genero_codigo=False
    alfabeto=[]
    probabilidades=[]
    for i in range(len(contextoActual)+1):
        aux=contextoActual[i:]
        if aux in contextos:                  
            if letra in contextos[aux]: #He encontrado la letra en el contexto particular
                if not(genero_codigo):#Si es la primera vez genero el código
                    genero_codigo=True
                    for aux_letra in contextos[aux]:
                        alfabeto.append(aux_letra)
                        probabilidades.append(contextos[aux][aux_letra])
                        tamanyo_del_contexto_y_letra=[len(aux),letra]
                    if debug:
                        print("Contexto ["+red(aux)+"] en el que se envía la letra "+red(letra))
                        print(alfabeto, probabilidades)                    
                contextos[aux][letra]+=1 #acualizo la frecuencia, después de generar el código, sino no se podría 'descodificar'
            else:
                contextos[aux][letra]=1 #contexto conocido, pero la letra no aparece en el contexto: la añada
        else:
            contextos[aux]={letra:1} #contexto nuevo, lo añado
            
    if not(genero_codigo): #es la primera vez que aparece la letra
        tamanyo_del_contexto_y_letra=['$',letra]
        if debug:
                print("Contexto ["+red(aux)+"] en el que se envía la letra "+red(letra))
                print(alfabeto, probabilidades) 
                              
                            
    if debug:
        print("Código",tamanyo_del_contexto_y_letra, alfabeto, probabilidades) 
             
    return contextos, tamanyo_del_contexto_y_letra, alfabeto,probabilidades


#%%

"""
--------------------------------------------------------
EJEMPLOS
--------------------------------------------------------
"""

mensaje='xyzzxyxyzza'.upper()
#mensaje="MISSISSIPPIMISSISSIPPIRIVER"
#mensaje=2*'La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte.'

print(mensaje)

orden=2
contextos, codigo=PPMCode(mensaje,orden,True)
print("\nLista final de items para pasar a codificación aritmética/huffman")
for k in range(len(codigo)):
    print(codigo[k])
#%%



#%%
##with open("la_regenta", "r",encoding='utf-8', errors='ignore') as myfile:
##with open("la_regenta", "r",encoding='iso8859', errors='replace') as myfile:
#with open("la_regenta", "r",encoding='iso8859') as myfile:
with open ("../standard_test_text/la_regenta_utf8", "r") as myfile:
    mensaje=myfile.read()
mensaje=mensaje[200:40000]   

orden=3
contextos, codigo=PPMCode(mensaje,orden,False)
for k in range(orden+1):
    print('---------- Contexto: ',k,'------------')
    aux={}
    for i in contextos:
        if len(i)==k: 
            aux.update({i:contextos[i]})
    aux=sorted(aux.items(), key=lambda x: x[0])
    for i in aux:
        print(i)
        
