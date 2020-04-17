# -*- coding: utf-8 -*-
"""

"""

'''
Dada la lista L de longitudes de las palabras de un código 
q-ario, decidir si pueden definir un código.

'''

def  kraft1(L, q=2):
    sum = 0
    for li in L:
        sum = sum + (1/pow(q,li))
    return sum <= 1



'''
Dada la lista L de longitudes de las palabras de un código 
q-ario, calcular el máximo número de palabras de longitud 
máxima, max(L), que se pueden añadir y seguir siendo un código.

'''

def  kraft2(L, q=2):
    sum = 0
    for li in L:
        sum = sum + (1/pow(q,li))
    n = 0
    maxnumber = max(L)
    while sum < 1:
        sum = sum + (1/pow(q,maxnumber))
        n = n + 1
    return n
'''
Dada la lista L de longitudes de las palabras de un  
código q-ario, calcular el máximo número de palabras 
de longitud Ln, que se pueden añadir y seguir siendo 
un código.
'''

def  kraft3(L, Ln, q=2):
    sum = 0
    for li in L:
        sum = sum + (1/pow(q,li))
    n = 0
    while sum < 1:
        sum = sum + (1/pow(q,Ln))
        n = n + 1
    return n

'''
Dada la lista L de longitudes de las palabras de un  
código q-ario, hallar un código prefijo con palabras 
con dichas longitudes

L=[2, 3, 3, 3, 4, 4, 4, 6]  
codigo binario (no es único): ['00', '010', '011', '100', '1010', '1011', '1100', '110100']

L=[1, 2, 2, 2, 3, 3, 5, 5, 5, 7]  
codigo ternario (no es único): ['0', '10', '11', '12', '200', '201', '20200', '20201', '20202', '2021000']
'''
def busqueda_recursiva(L,q, word):
    if L == []:
        return []
    else:
        if (L[0] == len(word)):
            i = len(word)-1
            backtracking = False
            while i >= 0 and not backtracking:
                if (ord(word[i])+1 - ord('0') < q):
                        backtracking = True
                else:
                    i = i-1
            return [word]+busqueda_recursiva(L[1:(len(L)+1)],q, word[0:i]+chr(ord(word[i])+1))
        else:
            return busqueda_recursiva(L,q,word+'0')
        
            
              
def Code(L,q=2):
    return busqueda_recursiva(sorted(L),q, '')



#%%

'''
Ejemplos
'''
#%%

L=[2,3,3,3,4,4,4,6]
q=2

print("\n",sorted(L),' codigo final:',Code(L,q))
print(kraft1(L,q))
print(kraft2(L,q))
print(kraft3(L,max(L)+1,q))

#%%
q=3
L=[1,3,5,5,3,5,7,2,2,2]
print(sorted(L),' codigo final:',Code(L,q))
print(kraft1(L,q))


#%%
'''
Dada la lista L de longitudes de las palabras de un  
código q-ario, hallar un código CANÓNICO binario con palabras 
con dichas longitudes
'''

def CodeCanonico(L):
    C = Code(L,2)
    R = []
    for l in L:
        for c in C:
            if (len(c) == l):
                R.append(c)
                C.remove(c)
    return R
        


#%%
'''
Ejemplos
'''
#%%
L=[3, 3, 3, 3,3, 2, 4, 4]
C=CodeCanonico(L)
print(C)
C==['010', '011', '100', '101', '110', '00', '1110', '1111']

#%%
L=[7,2,3,3,3,3,23,4,4,4,6]
C=CodeCanonico(L)
print(C)
C==['1111010', '00', '010', '011', '100', '101', '11110110000000000000000', '1100', '1101', '1110', '111100']


