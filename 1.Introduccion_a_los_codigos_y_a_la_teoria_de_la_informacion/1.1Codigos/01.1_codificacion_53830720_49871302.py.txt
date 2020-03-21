# -*- coding: utf-8 -*-


'''
0. Dada una codificación R, construir un diccionario para codificar m2c y 
otro para decodificar c2m
'''
R = [('a','0'), ('b','11'), ('c','100'), ('d','1010'), ('e','1011')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])


'''
1. Definir una función Encode(M, m2c) que, dado un mensaje M y un diccionario 
de codificación m2c, devuelva el mensaje codificado C.
'''

def Encode(M, m2c):
    i = 0
    V = ''
    while i < len(M):
        j = 1
        while j < len(M):
            if (M[i:(i+j)] in m2c):
                V = V + m2c[M[i:(i+j)]]
                i = i+j
                break
            else:
                j = j + 1
    return V
    

''' 
2. Definir una función Decode(C, c2m) que, dado un mensaje codificado C y un diccionario 
de decodificación c2m, devuelva el mensaje original M.
'''
def Decode(C,c2m):
    i = 0
    V = ''
    while i < len(C):
        j = 1
        while j < len(C):
            if (C[i:(i+j)] in c2m):
                V = V + c2m[C[i:(i+j)]]
                i = i+j
                break
            else:
                j = j + 1
    return V
  

#------------------------------------------------------------------------
# Ejemplo 1
#------------------------------------------------------------------------

R = [('a','0'), ('b','11'), ('c','100'), ('d','1010'), ('e','1011')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])
M='aabacddeae'
C=Encode(M,m2c)
print(C=='0011010010101010101101011')
print(M==Decode(C,c2m))



#------------------------------------------------------------------------
# Ejemplo 2
#------------------------------------------------------------------------
R = [('a','0'), ('b','10'), ('c','110'), ('d','1110'), ('e','1111')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])
M='aabacddeaeabc'
C=Encode(M,m2c)
print(C=='0010011011101110111101111010110')
print(M==Decode(C,c2m))




#------------------------------------------------------------------------
# Ejemplo 3
#------------------------------------------------------------------------
R = [('ab','0'), ('cb','11'), ('cc','100'), ('da','1010'), ('ae','1011')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in m2c.items()])

M='ababcbccdaae'
C=Encode(M,m2c)
print(C=='001110010101011')

print(M==Decode(C,c2m))


''' 
3.
Codificar y decodificar 20 mensajes aleatorios de longitudes también aleatorias.  
Comprobar si los mensajes decodificados coinciden con los originales.
'''




#------------------------------------------------------------------------
# Ejemplo 4
#------------------------------------------------------------------------
R = [('a','0'), ('b','01'), ('c','011'), ('d','0111'), ('e','1111')]

# encoding dictionary
m2c = dict(R)

# decoding dictionary
c2m = dict([(c,m) for m, c in R])


''' 
4. Codificar y decodificar los mensajes  'ae' y 'be'. 
Comprobar si los mensajes decodificados coinciden con los originales.
'''
C = Encode('ae', m2c)
C2 = Encode('be', m2c)
print(Decode(C,c2m))
print(Decode(C2,c2m))



'''
¿Por qué da error?
Esto se debe a que no es un codigo prefijo. La palabra 0 es prefijo d 01, 011, 01111. La palabra 01 es prefijo de 011 ...
(No es necesario volver a implementar Decode(C, m2c) para que no dé error)
'''



  




