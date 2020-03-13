# -*- coding: utf-8 -*-
"""

"""

import math
import resource, sys
import time

#%%----------------------------------------------------

'''
Dada una distribucion de probabilidad, hallar un código canónico de Huffman asociado
'''


iteracion = 0

resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

def generacion_arbol(codigo):
    if (len(codigo) == 1):
        return codigo
    else:
        print("error ordenacion")
        global iteracion 
        iteracion = iteracion +1
        print(iteracion)
        if iteracion > 16882:
            print(codigo)
        codigo.sort()
        print("error ordenacion 2")
        suma_probabilidades = round(codigo[0][0] + codigo[1][0],10)
        return generacion_arbol( [ [suma_probabilidades, codigo[0], codigo[1]] ] + codigo[2:len(codigo)] )

def recorrer_arbol(codigos, c):
    if (len(codigos) == 1):
        return [ [codigos[0],c]]
    else:
        return recorrer_arbol(codigos[1], c+ '1') + recorrer_arbol(codigos[2], c + '0')
    

def Huffman(p):
    if len(p) == 1:
        return [[1, '0']]
    codigos = []
    for probabilidad in p:
        codigos = codigos + [[probabilidad]]
    codigos = generacion_arbol(codigos)
    codigo_huffman = recorrer_arbol(codigos[0],'')
    return codigo_huffman

p1=[0.05, 0.5, 0.1, 0.25, 0.05, 0.05]
p2=[0.102, 0.106, 0.053, 0.114, 0.0081, 0.106, 0.0088, 0.030, 0.056, 0.055, 0.032, 0.094, 0.0075, 0.078, 0.1496]

#print(Huffman(p1))
#print(Huffman(p2))
'''
p=[0.05, 0.5, 0.1, 0.25, 0.05, 0.05]
Huffman(p)=['11110', '0', '110', '10', '11111', '1110'] (¡NO ES ÚNICO!)

p=[0.102, 0.106, 0.053, 0.114, 0.0081, 0.106, 0.0088, 0.030, 0.056, 0.055, 0.032, 0.094, 0.0075, 0.078, 0.1496]
Huffman(p)=['000', '001', '11100', '010', '1111110', '011', '111110', '11101', '1010', '1011', '11110', '1100', '1111111', '1101', '100']
'''



#%%----------------------------------------------------

'''
Dada la ddp p=[1/n,..../1/n] con n=2**8, hallar un código de Huffman asociado,
la entropía de p y la longitud media de código de Huffman hallado.
'''
def H1(n):
    suma = 0
    for p in n:
        if (p > 0):
            suma = suma+ (math.log(p,2))*p
    return -suma
        
def LongitudMedia(C,p):
    suma = 0
    for i in range(0,len(C)):
        suma = suma + len(C[i])*p[i]
    return suma

n=2**8
p=[1/n for _ in range(n)]
#huffman = Huffman(p)
#C = [c for [_,c] in huffman]

#print(huffman)
#print(H1(p))
#print(LongitudMedia(C,p))




#%%----------------------------------------------------

'''
Dado un mensaje hallar la tabla de frecuencia de los los grupos formados 
por "numero_de_simbolos" símbolos que lo componen.

Si len(mensaje)%numero_de_simbolos!=0, se completa el mensaje con el último 
símbolo hasta que la longitud de mensaje es un múltiplo de "numero_de_simbolos"

'''


def tablaFrecuencias(mensaje,numero_de_simbolos=1):
    frecuencias = dict([])
    for i in range(0,len(mensaje),numero_de_simbolos):
        if (i+numero_de_simbolos-1<len(mensaje)):
            simbolo = mensaje[i:(i+numero_de_simbolos)] 
            if simbolo in frecuencias:
                frecuencias[simbolo] = frecuencias[simbolo] +1
            else:
                frecuencias[simbolo] = 1
        else:
                frecuencias[mensaje[i:len(mensaje)]] = 1
    lista = list(map(list, frecuencias.items()))
    return lista


mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por un instinto de prudencia y armonía que modificaba las vulgares exageraciones de esta arquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababa en pararrayos.'
list1 = tablaFrecuencias(mensaje, numero_de_simbolos=1) 
list2 = [[' ', 384], [',', 46], ['.', 8], [';', 3], ['B', 1], ['C', 2], ['E', 2], ['L', 3], ['N', 1], ['S', 2], ['V', 1], ['a', 267], ['b', 46], ['c', 64], ['d', 104], ['e', 238], ['f', 6], ['g', 23], ['h', 15], ['i', 98], ['j', 7], ['l', 112], ['m', 49], ['n', 111], ['o', 141], ['p', 48], ['q', 25], ['r', 126], ['s', 145], ['t', 68], ['u', 87], ['v', 11], ['x', 1], ['y', 30], ['z', 11], ['á', 11], ['é', 3], ['í', 12], ['ñ', 4], ['ó', 5], ['ú', 1]]
list1.sort()
list2.sort()
#print(list1 == list2)
list1 = tablaFrecuencias(mensaje, numero_de_simbolos=2)
list2 = [[' C', 1], [' L', 1], [' N', 1], [' a', 17], [' b', 5], [' c', 18], [' d', 22], [' e', 20], [' f', 2], [' g', 2], [' h', 6], [' i', 4], [' j', 2], [' l', 15], [' m', 7], [' n', 3], [' o', 4], [' p', 16], [' q', 7], [' r', 5], [' s', 8], [' t', 2], [' u', 6], [' v', 1], [' y', 8], [' á', 1], [', ', 17], [',d', 1], ['. ', 4], ['.C', 1], ['; ', 1], ['Ba', 1], ['El', 1], ['En', 1], ['La', 2], ['Sa', 1], ['Su', 1], ['Ve', 1], ['a ', 39], ['a,', 7], ['a.', 1], ['ab', 11], ['ac', 6], ['ad', 7], ['ag', 1], ['ai', 1], ['al', 9], ['am', 1], ['an', 9], ['ap', 2], ['aq', 2], ['ar', 9], ['as', 10], ['at', 3], ['añ', 1], ['ba', 12], ['bl', 1], ['br', 4], ['ca', 6], ['ce', 2], ['ci', 7], ['co', 5], ['cr', 2], ['ct', 1], ['cu', 3], ['cí', 1], ['d ', 1], ['d,', 1], ['da', 7], ['de', 24], ['di', 4], ['do', 11], ['dr', 4], ['du', 1], ['e ', 35], ['e,', 3], ['ea', 1], ['eb', 2], ['ed', 3], ['ei', 1], ['el', 11], ['em', 2], ['en', 17], ['ep', 3], ['eq', 1], ['er', 9], ['es', 20], ['et', 1], ['ev', 2], ['ez', 2], ['eñ', 1], ['fa', 1], ['fl', 1], ['fu', 1], ['ga', 7], ['ge', 1], ['gi', 1], ['gl', 1], ['go', 1], ['gr', 2], ['gu', 3], ['ha', 5], ['hi', 1], ['ho', 2], ['hu', 1], ['ia', 3], ['ib', 2], ['ic', 3], ['id', 4], ['ie', 10], ['if', 1], ['ig', 2], ['il', 2], ['im', 1], ['in', 2], ['io', 3], ['ir', 2], ['is', 5], ['it', 1], ['iu', 2], ['iz', 1], ['ié', 1], ['ja', 4], ['l ', 10], ['l,', 1], ['la', 14], ['lc', 1], ['le', 10], ['lg', 1], ['li', 4], ['ll', 6], ['lo', 9], ['lt', 1], ['lu', 1], ['ma', 8], ['mb', 2], ['me', 2], ['mi', 2], ['mn', 1], ['mo', 7], ['mp', 2], ['mu', 1], ['má', 2], ['n ', 13], ['na', 7], ['nc', 4], ['nd', 8], ['ne', 2], ['ng', 1], ['ni', 1], ['nn', 1], ['no', 6], ['nq', 1], ['ns', 2], ['nt', 8], ['nu', 1], ['nv', 1], ['nz', 1], ['nó', 1], ['o ', 21], ['o,', 6], ['o;', 1], ['ob', 1], ['oc', 1], ['od', 4], ['oe', 1], ['oi', 1], ['ol', 5], ['om', 6], ['on', 4], ['op', 1], ['or', 8], ['os', 7], ['ot', 1], ['pa', 6], ['pe', 8], ['pi', 3], ['pl', 1], ['po', 3], ['pr', 2], ['pu', 1], ['qu', 11], ['r ', 5], ['r,', 2], ['ra', 18], ['rb', 1], ['rc', 1], ['rd', 1], ['re', 10], ['ri', 5], ['rm', 2], ['ro', 6], ['rq', 1], ['rr', 4], ['rs', 3], ['rt', 1], ['ru', 4], ['rv', 1], ['rá', 1], ['s ', 34], ['s,', 8], ['s.', 2], ['sa', 3], ['sb', 2], ['sc', 3], ['se', 10], ['si', 4], ['sm', 1], ['so', 5], ['sq', 2], ['st', 5], ['su', 4], ['sí', 1], ['ta', 12], ['te', 9], ['ti', 6], ['to', 5], ['tr', 7], ['tu', 3], ['u ', 1], ['ua', 1], ['ub', 2], ['ud', 1], ['ue', 11], ['ui', 6], ['uj', 1], ['ul', 1], ['um', 1], ['un', 8], ['ur', 2], ['us', 3], ['ut', 1], ['uy', 1], ['ve', 1], ['vi', 2], ['vo', 1], ['vu', 2], ['xa', 1], ['y ', 15], ['ya', 1], ['ye', 2], ['yo', 3], ['z ', 2], ['za', 2], ['zo', 1], ['zu', 1], ['zá', 1], ['á ', 1], ['ám', 1], ['án', 3], ['ás', 1], ['é;', 1], ['és', 1], ['í ', 1], ['ía', 7], ['ín', 2], ['ña', 1], ['ño', 1], ['ób', 1], ['ón', 2], ['ót', 1], ['ús', 1]]
list1.sort()
list2.sort()
#print(list1 == list2)



'''
mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por un instinto de prudencia y armonía que modificaba las vulgares exageraciones de esta arquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababa en pararrayos.'

tablaFrecuencias(mensaje, numero_de_simbolos=1)=[[' ', 384], [',', 46], ['.', 8], [';', 3], ['B', 1], ['C', 2], ['E', 2], ['L', 3], ['N', 1], ['S', 2], ['V', 1], ['a', 267], ['b', 46], ['c', 64], ['d', 104], ['e', 238], ['f', 6], ['g', 23], ['h', 15], ['i', 98], ['j', 7], ['l', 112], ['m', 49], ['n', 111], ['o', 141], ['p', 48], ['q', 25], ['r', 126], ['s', 145], ['t', 68], ['u', 87], ['v', 11], ['x', 1], ['y', 30], ['z', 11], ['á', 11], ['é', 3], ['í', 12], ['ñ', 4], ['ó', 5], ['ú', 1]]

tablaFrecuencias(mensaje, numero_de_simbolos=2)=[[' C', 1], [' L', 1], [' N', 1], [' a', 17], [' b', 5], [' c', 18], [' d', 22], [' e', 20], [' f', 2], [' g', 2], [' h', 6], [' i', 4], [' j', 2], [' l', 15], [' m', 7], [' n', 3], [' o', 4], [' p', 16], [' q', 7], [' r', 5], [' s', 8], [' t', 2], [' u', 6], [' v', 1], [' y', 8], [' á', 1], [', ', 17], [',d', 1], ['. ', 4], ['.C', 1], ['; ', 1], ['Ba', 1], ['El', 1], ['En', 1], ['La', 2], ['Sa', 1], ['Su', 1], ['Ve', 1], ['a ', 39], ['a,', 7], ['a.', 1], ['ab', 11], ['ac', 6], ['ad', 7], ['ag', 1], ['ai', 1], ['al', 9], ['am', 1], ['an', 9], ['ap', 2], ['aq', 2], ['ar', 9], ['as', 10], ['at', 3], ['añ', 1], ['ba', 12], ['bl', 1], ['br', 4], ['ca', 6], ['ce', 2], ['ci', 7], ['co', 5], ['cr', 2], ['ct', 1], ['cu', 3], ['cí', 1], ['d ', 1], ['d,', 1], ['da', 7], ['de', 24], ['di', 4], ['do', 11], ['dr', 4], ['du', 1], ['e ', 35], ['e,', 3], ['ea', 1], ['eb', 2], ['ed', 3], ['ei', 1], ['el', 11], ['em', 2], ['en', 17], ['ep', 3], ['eq', 1], ['er', 9], ['es', 20], ['et', 1], ['ev', 2], ['ez', 2], ['eñ', 1], ['fa', 1], ['fl', 1], ['fu', 1], ['ga', 7], ['ge', 1], ['gi', 1], ['gl', 1], ['go', 1], ['gr', 2], ['gu', 3], ['ha', 5], ['hi', 1], ['ho', 2], ['hu', 1], ['ia', 3], ['ib', 2], ['ic', 3], ['id', 4], ['ie', 10], ['if', 1], ['ig', 2], ['il', 2], ['im', 1], ['in', 2], ['io', 3], ['ir', 2], ['is', 5], ['it', 1], ['iu', 2], ['iz', 1], ['ié', 1], ['ja', 4], ['l ', 10], ['l,', 1], ['la', 14], ['lc', 1], ['le', 10], ['lg', 1], ['li', 4], ['ll', 6], ['lo', 9], ['lt', 1], ['lu', 1], ['ma', 8], ['mb', 2], ['me', 2], ['mi', 2], ['mn', 1], ['mo', 7], ['mp', 2], ['mu', 1], ['má', 2], ['n ', 13], ['na', 7], ['nc', 4], ['nd', 8], ['ne', 2], ['ng', 1], ['ni', 1], ['nn', 1], ['no', 6], ['nq', 1], ['ns', 2], ['nt', 8], ['nu', 1], ['nv', 1], ['nz', 1], ['nó', 1], ['o ', 21], ['o,', 6], ['o;', 1], ['ob', 1], ['oc', 1], ['od', 4], ['oe', 1], ['oi', 1], ['ol', 5], ['om', 6], ['on', 4], ['op', 1], ['or', 8], ['os', 7], ['ot', 1], ['pa', 6], ['pe', 8], ['pi', 3], ['pl', 1], ['po', 3], ['pr', 2], ['pu', 1], ['qu', 11], ['r ', 5], ['r,', 2], ['ra', 18], ['rb', 1], ['rc', 1], ['rd', 1], ['re', 10], ['ri', 5], ['rm', 2], ['ro', 6], ['rq', 1], ['rr', 4], ['rs', 3], ['rt', 1], ['ru', 4], ['rv', 1], ['rá', 1], ['s ', 34], ['s,', 8], ['s.', 2], ['sa', 3], ['sb', 2], ['sc', 3], ['se', 10], ['si', 4], ['sm', 1], ['so', 5], ['sq', 2], ['st', 5], ['su', 4], ['sí', 1], ['ta', 12], ['te', 9], ['ti', 6], ['to', 5], ['tr', 7], ['tu', 3], ['u ', 1], ['ua', 1], ['ub', 2], ['ud', 1], ['ue', 11], ['ui', 6], ['uj', 1], ['ul', 1], ['um', 1], ['un', 8], ['ur', 2], ['us', 3], ['ut', 1], ['uy', 1], ['ve', 1], ['vi', 2], ['vo', 1], ['vu', 2], ['xa', 1], ['y ', 15], ['ya', 1], ['ye', 2], ['yo', 3], ['z ', 2], ['za', 2], ['zo', 1], ['zu', 1], ['zá', 1], ['á ', 1], ['ám', 1], ['án', 3], ['ás', 1], ['é;', 1], ['és', 1], ['í ', 1], ['ía', 7], ['ín', 2], ['ña', 1], ['ño', 1], ['ób', 1], ['ón', 2], ['ót', 1], ['ús', 1]]
'''

#%%----------------------------------------------------
'''
Definir una función que codifique un mensaje utilizando un código de Huffman 
obtenido a partir de las frecuencias de los caracteres del mensaje.

Definir otra función que decodifique los mensajes codificados con la función 
anterior.
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

def obtener_m2c(huffman,tablaFrecuencias):
    m2c = []
    for c in huffman:
        for p in tablaFrecuencias:
            if (c[0] == p[1]):
                m2c = m2c + [(p[0],c[1])]
                tablaFrecuencias.remove(p)
                break
    return m2c
    
def EncodeHuffman(mensaje_a_codificar,numero_de_simbolos=1):
    t0 = time.time()
    longitud_mensaje = len(mensaje_a_codificar)
    residuo = longitud_mensaje-((int(longitud_mensaje/numero_de_simbolos)+1)*numero_de_simbolos);
    if longitud_mensaje% numero_de_simbolos != 0:
        for i in range(0,residuo):
            mensaje_a_codificar = mensaje_a_codificar + '.'
    tablafrecuencias = tablaFrecuencias(mensaje_a_codificar, numero_de_simbolos)
    p = [probabilidad for [_, probabilidad] in tablafrecuencias]
    huffman  = Huffman(p)
    m2c = obtener_m2c(huffman,tablafrecuencias)
    mensaje_codificado = Encode(mensaje_a_codificar,dict(m2c))
    longitud_mensaje = len(mensaje_codificado)
    return mensaje_codificado, m2c, longitud_mensaje
    
    
def DecodeHuffman(mensaje_codificado,m2c,longitud_mensaje):
    c2m = dict([(c,m) for (m,c) in m2c])
    mensaje_decodificado = Encode(mensaje_codificado,c2m)
    mensaje_decodificado = mensaje_decodificado[0:longitud_mensaje]
    return mensaje_decodificado
        

mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por un instinto de prudencia y armonía que modificaba las vulgares exageraciones de esta arquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababa en pararrayos.'

# mensaje_codificado, m2c, longitud_mensaje = EncodeHuffman(mensaje,numero_de_simbolos=1)
# print(DecodeHuffman(mensaje_codificado,m2c,longitud_mensaje) )


"""
Ejemplo

mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por un instinto de prudencia y armonía que modificaba las vulgares exageraciones de esta arquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababa en pararrayos.'

EncodeHuffman(mensaje,numero_de_simbolos=1)

¡ATENCIÓN: m2c NO ES ÚNICO!
m2c=
Diccionario ordenado alfabéticamente:  [(' ', '000'), (',', '111000'), ('.', '11110110'), (';', '1111111000'), ('B', '11111111100'), ('C', '1111111001'), ('E', '1111111010'), ('L', '1111111011'), ('N', '11111111101'), ('S', '1111111100'), ('V', '111111111110'), ('a', '001'), ('b', '111001'), ('c', '10110'), ('d', '10111'), ('e', '010'), ('f', '11110111'), ('g', '1111000'), ('h', '1111001'), ('i', '11000'), ('j', '11111000'), ('l', '0110'), ('m', '11001'), ('n', '0111'), ('o', '1000'), ('p', '111010'), ('q', '1111010'), ('r', '1001'), ('s', '1010'), ('t', '11010'), ('u', '11011'), ('v', '11111001'), ('x', '111111111111'), ('y', '111011'), ('z', '11111010'), ('á', '11111011'), ('é', '1111111101'), ('í', '11111100'), ('ñ', '111111010'), ('ó', '111111011'), ('ú', '11111111110')]
Diccionario ordenado longitud código:  [(' ', '000'), ('a', '001'), ('e', '010'), ('l', '0110'), ('n', '0111'), ('o', '1000'), ('r', '1001'), ('s', '1010'), ('c', '10110'), ('d', '10111'), ('i', '11000'), ('m', '11001'), ('t', '11010'), ('u', '11011'), (',', '111000'), ('b', '111001'), ('p', '111010'), ('y', '111011'), ('g', '1111000'), ('h', '1111001'), ('q', '1111010'), ('.', '11110110'), ('f', '11110111'), ('j', '11111000'), ('v', '11111001'), ('z', '11111010'), ('á', '11111011'), ('í', '11111100'), ('ñ', '111111010'), ('ó', '111111011'), (';', '1111111000'), ('C', '1111111001'), ('E', '1111111010'), ('L', '1111111011'), ('S', '1111111100'), ('é', '1111111101'), ('B', '11111111100'), ('N', '11111111101'), ('ú', '11111111110'), ('V', '111111111110'), ('x', '111111111111')]
{'o': '1110', 'v': '11011101', 'c': '11111', 'j': '00001110', ';': '1011011101', 'u': '11010', 'V': '101101110011', 'r': '0010', 'b': '101110', 'á': '11011001', 'E': '1101111111', 'x': '101101110010', 'g': '1101101', 's': '1100', 'S': '1101111110', 'ó': '110111001', 'e': '011', 'B': '11011111000', 'm': '00000', 'l': '0011', ',': '101111', 'z': '11011000', ' ': '100', 'd': '10101', 'p': '101100', 'L': '1011011111', 'í': '10110110', 'C': '1101111101', 'q': '1011010', 'N': '11011111001', '.': '11011110', 'i': '10100', 'a': '010', 'y': '000010', 'ñ': '110111000', 'ú': '10110111000', 't': '11110', 'h': '0000110', 'é': '1011011110', 'n': '0001', 'f': '00001111'}

mensaje_codificado='111111101100100011110010101001100011000101100010001011011000110111011100110111000101111000100111001111111000010000110001000101011000010101011010001111101100001111111010011000011111001110000100111110101000000111111110011011100111100000010110001011011000010011111010010000111011000111010010100101011111010100010101000111000000010110011110101101111111000001111001001000011000110100000111110111110010101010000111001011000101111111010110110101011011000011100110100001111010110110100001010010000100100110101111000001111001001011100000101100001011010001001100101010010001111001001101101100000100001001100001111111110110001001110100101111011000011111110100111000011000110100001011000101100110010101000001111000000111100100111100111111100001000110011111101110100001001110111100010111100000011110101101101000001001100001001110111100110001001000010101011010100111000101110100111110100100001011101000001101000101000010010101100110000110110000111100010100001011101000011101010000110111110011000111000000110101001001111010100010101110000001110100011111100000110100001110110001110100011110100100110010101000011110101101101000011000111001001011100010111010000001100110011000111011100000001001110000011001100110001110111000111000000101110100000011011001010010010000100111000001101100101001001111000000101110100000101010111101011011110000111001000010011100001010101111010110111100001110010001001010111110011000011000101111011110000001110110001110100101001101011000111100011011110001111111101011110111100010100101110000001011010001100110000001100100110011100011101010001010001101000011110101101101000010100100001110011101110101011000101110001110110001111001110111110110100111000111011000111101011011010000010011000000111000100101000001001111111100111011010011011111001010000010011100010101101110100001110100110110000101111000110110101010000110000111111110011100010101100011100101100101010111101100001111111001110110010110000110101101110011110010011010000101110100001110101100001100110110110100110100010101110000000011111010110110100110011000110100001100111000111100000111111000001101000010111010000011000100011100100110101101110010011110000000011111010110110100110011000110100001010100011100110010011010000101110100001101010001011110000001010010000111110001101101111101000111100100101110000100111000110110111000110011000011111010111111011011111100000011101000110011111101111100100101111010010000101101000110011000000101111000100111001110001011100110100001101101110001100110001100101001111101010000001110110001110011001110000111101100011110010010111000101110100000111110110101111100110000001010100011100110010101010001011011010001101110011010111000000101111100010101110100101001101011111011011110111100010100101110000001101010010101110100010111101111000000110110111001101000011101010001001000011000110100001110100011001010101110101010000111100100110101101000100001101000101000010110100111000101011010001011001010100001101001011001111001011010001001100010101000101000010111010000011010001010000111101110011001100001100101010111000000100011010100100110100001111001001101011010001000011010001010000101100011001110100100110010101000010111010000111010001111010010011000011001001011000011101001011110000011011110000000010000110001101000001010101111010110111100001110011010111000000111011000111100100111100111111100001000111010011011011110010010001111010110110100000110011001011110000011110010010000010001101101110001101001010011011001010010001110101100010101000111000000111011000001100101001111100001100110001000111101011011010000101001000011000011110110100111011101011010001111001001000111010001100100100010111111111000011010111000000100000011101000110010010000011111110101000101011100000001001110000110001000111110011100010111100111000010100100100010111010000110110111000010101010110001111010001100100111010010111000000001111100000110011001001101110010000010001101101110001110100110100011001100011110110000111111111110010110101101110101101000111100000001100010001100111011111011000011110001110010110010000111011000011001000101100001011011000110111011100110111111000000101101000100111010010000010011100001100101111100000101111000000101011000111100001101000111000000111100100110110111111000010000110001000101111100011110000101010110101100011111101101110001011101001100001011010001011011000101111000000111011000101110100000110001000100001100110001000111010100010111100111000101110011110000001110110001011101010101011000101111010001111001001000100011101101001111011110000000100111110101001010000101011011010111111010100010100000100110000110011000011111111101111010100001111000000111011000111101110011100111000011011000001100100011111010110111100111100111000101111000000101110100000110001000101100011100111101000101110010001011101000010110100010011000111000000111101011011010000100101011010110111100111100100111100100100000101100110111110110000100111000011010000000010110110101000000101110100000110001000010101011100101001101101000111010100010011001010000010011100001100010001111111100001011111010001000111111111000011010111111000110110001011000111110110000111111101100100011010100010011001010000101110100000110001000101100011101001010111100100101101110000001110101000010110010010001001100011001111110110111110101100010110100000010111010000111010110000101011110010011110001011101001101100010110001101111000000111100111000110010111100011100000010111010000101111101101101011001010100000110111111000111010001101000010111010000111001010011001100101111101000100011001110111011100100011101100011101001010010100111011101011100000001010010010001000111001100100100010111010011000010101100011110000110100000010111110000101111101000011101100010100101100010101110000000011101101111111010110110100000010111110100101010000101101000110010100111111110100011011100111100000010111010000010101011010110000110100000011110001111110111101011000101101000111000000111010010100110001110000001011000111100101000010111010101101100010011110000001100110001011101010010011011110000001110101000100100011011011100011000011110101101011000011111010100000010111010000111010100111011101110100111101101100000100011101100000110011100110000111111111000010001111010110110100001100110001011111000111101111100010110001111001001000011000110100001111100111011011011110000011001010101000001011111111111100111110000101001001101101100010000111010101000010111010000010101011010001000001100111110101101111000110100101011011010110111001001111101100001111111011001000111110011100010101101000100001111000000101001000011110111001110101100011110000011110010010001011010000111110100101100111101001100010111101111000000111100110001001001101000011101100011110011000100100110100000011111010110110100110000111111000111101111100010110010000101110101110101100001010111100100100011110101101101000010100101111110100010110001111001001000001011000010110110000100110100011111110000000111100000001010010010001101101110010001011101000001010100011010000110101000100110010101010000101101101111101100100000111110001101111111000001000101001000011110101101111000010111001100100100010111010101011011110101100001101110000001100111111011101000011110111011000110110001101000011110101101101000001010101110010100110110100011010111000000001110010010111010100100110111001101011100000010110100011001100000010100101111110101000100111000110100011010000101101101110011010110001010000111101011011010000001111010100111000010110100010111101110101100100110101100000110111100000001001100001011010001001101011111111011111111000000010100100100011001001101101100011111010001000101011000011100011101001010011011101010010000111001101110010001011101000010101101100001010101110101100010011100011010110110010110000111100010010010111101110101111101000111100000011101100011110010011010110100010001010110111010101001011110001101101111011110001010000101101000100110010101011110001001010101011100000001001100101111000001011111010010000111001001011000111011101011010100100110111001111000000101011011111001111111000010001011010001100110000001111011111011010100111010010000101100011010110101100001100110100011100000001100010111111110101111101101111011110001010010101110101010101110100000010110011011111100000010011100011101011000100111111011110011100010111010000101110100001111101101111111000110110110100000011110001001001101101100010001010100011100000011000011111000110011100011010001111001011001000001001110001010110111010000110010101011111000101110011010000111011000111010100110001110101000100110110110001000011101010101111011011111110011000110011000000111100100111111010000101110100001100111111111110101010110110110110100010100001110110000111010100111111001110001000101000001100010001110101100001010111100100100001001111001100010101011011111011011110111100010100100000100111000011000100011101011000010101111001001000110101001010111010001111001001000001000011000100000101101101011011100100111100000011110010011011011000010011110111100000001011110101101111000011011000111001100111000100010100001011101000000110110100111111101111100100111010001000010011100001001100000011100010010101111111000000111011000101101000110011000000111010100110001011111000111100011000100000010111010000111110001101101011110001000101011001001011000111100100110010101010111000000010011100011011011100100011101011011011111010001000101110100001011000101101100011111010001000101001000011001001011111010010011111111100001111000000101101101100101100001100011001001011111010001101110011110000001101101110010001110011000011000100011110001001001011110111010000101110101110011001100001111011001000010111100010010011011110001110000001110110000100111101101100011001001000100011010100100100011001111110111010000111010010111101011011010011111101100111100000011101100010101000111001100101000011111111011010110100010001101101110010001011010011101111111010000101110100001111001110000101001100110000001111010110110100000011011000111100100111100100100001001110001110100011001001100110010011110111000101011110110'

longitud_mensaje=9834
"""




'''
Si no tenemos en cuenta la memoria necesaria para almacenar el diccionario, 
haced una estimación de la ratio de compresión para el fichero "la_regenta"
que encontraréis en Atenea con numero_de_simbolos=1
Si tenemos en cuenta la memoria necesaria para almacenar el diccionario, 
haced una estimación de la ratio de compresión.

Repetid las estimaciones con numero_de_simbolos=2,3,...
'''

with open ("la_regenta.txt", "r") as myfile:
    mensaje=myfile.read()
    EncodeHuffman(mensaje,4)
"""
n = 5
with open ("la_regenta.txt", "r") as myfile:
    mensaje=myfile.read()
    tamaño_original = len(mensaje)
    for numero_de_simbolos in range(1,n+1):
        print(numero_de_simbolos)
        _ , _, tamaño_comprimido = EncodeHuffman(mensaje,numero_de_simbolos)
        print(tamaño_original*8/tamaño_comprimido)
"""
    
    



