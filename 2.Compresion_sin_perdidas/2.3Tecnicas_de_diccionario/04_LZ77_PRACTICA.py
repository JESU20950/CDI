# -*- coding: utf-8 -*-


"""
Dado un mensaje, el tamaño de la ventana de trabajo L, y el tamaño
del buffer de búsqueda S dar la codificación del mensaje usando el
algoritmo LZ77


mensaje='cabracadabrarrarr'
LZ77Code(mensaje,12,6)=[['c', 0, 0], ['a', 0, 0],  ['b', 0, 0],
 ['r', 0, 0],  ['c', 1, 3],  ['d', 1, 2], ['r', 4, 7],  ['EOF', 4, 3]]
  
  
[símbolo, longitud_máxima_cadena, posición]    
"""

def LZ77Code(mensaje,S=12,L=6):
    #S is search buffer
    #L is the look-ahead buffer 
    code = []
    start_look_ahead = 0
    while (start_look_ahead < len(mensaje)):
        
        #find biggest preffix from the the look-ahead buffer that is in the search buffer
        finded_prefix = True
        longitud_maxima_cadena = 0
        iterator_search_buffer_final = 0
        begginning_search_buffer = start_look_ahead-S
        
        if (begginning_search_buffer < 0):
            begginning_search_buffer = 0
        end_look_ahead = start_look_ahead+L
        if (end_look_ahead > len(mensaje)):
            end_look_ahead = len(mensaje)
            
        while (start_look_ahead+longitud_maxima_cadena< end_look_ahead and finded_prefix): 
            finded_prefix = False
            iterator_search_buffer = start_look_ahead-1
            while (iterator_search_buffer >=  begginning_search_buffer):
                if(mensaje[iterator_search_buffer:(iterator_search_buffer+longitud_maxima_cadena)+1] == mensaje[start_look_ahead:(start_look_ahead+longitud_maxima_cadena)+1]):
                    finded_prefix = True
                    longitud_maxima_cadena = longitud_maxima_cadena +1
                    iterator_search_buffer_final = iterator_search_buffer
                    break
                
                iterator_search_buffer = iterator_search_buffer-1
                
        #insert the length of the prefix and consecutive char from the prefix
        if (start_look_ahead+longitud_maxima_cadena < len(mensaje)):
            if (longitud_maxima_cadena == 0):
                code = code + [[mensaje[start_look_ahead],0, 0]]
            else:
                code = code + [[mensaje[start_look_ahead+longitud_maxima_cadena],longitud_maxima_cadena, start_look_ahead-iterator_search_buffer_final]]
        else:
            code = code + [['EOF',longitud_maxima_cadena, start_look_ahead-iterator_search_buffer_final]]
            
        start_look_ahead = start_look_ahead+longitud_maxima_cadena+1
    return code



mensaje='patadecabra'
print(LZ77Code(mensaje,12,6))
mensaje='cabracadabrarrarr'
print(LZ77Code(mensaje,12,6))
    


"""
Dado un mensaje codificado con el algoritmo LZ77 hallar el mensaje 
correspondiente 

code=[['p', 0, 0], ['a', 0, 0],  ['t', 0, 0],  ['d', 1, 2],  ['e', 0, 0],
 ['c', 0, 0], ['b', 1, 4],  ['r', 0, 0], ['EOF', 1, 3]]
 
LZ77Decode(code)='patadecabra'

"""   

def getprefix(mensaje,c):
    longitud_maxima_cadena = c[1]
    position = len(mensaje)-c[2]
    while (longitud_maxima_cadena>0):
        mensaje = mensaje+ mensaje[position]
        position = position+1
        longitud_maxima_cadena = longitud_maxima_cadena-1
    return mensaje

def LZ77Decode(codigo):
    mensaje = ''
    for c in codigo:
        mensaje = getprefix(mensaje,c)
        if (c[0] != 'EOF'):
            mensaje = mensaje + c[0]
    return mensaje

code=[['p', 0, 0], ['a', 0, 0],  ['t', 0, 0],  ['d', 1, 2],  ['e', 0, 0],
 ['c', 0, 0], ['b', 1, 4],  ['r', 0, 0], ['EOF', 1, 3]]
 
print(LZ77Decode(code))
code = [['c', 0, 0], ['a', 0, 0],  ['b', 0, 0],
 ['r', 0, 0],  ['c', 1, 3],  ['d', 1, 2], ['r', 4, 7],  ['EOF', 4, 3]]
print(LZ77Decode(code))


"""
Jugar con los valores de S y L (bits_o y bits_l)
para ver sus efectos (tiempo, tamaño...)
"""


mensaje='La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos. La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos. La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos. La heroica ciudad dormía la siesta. El viento Sur, caliente y perezoso, empujaba las nubes blanquecinas que se rasgaban al correr hacia el Norte. En las calles no había más ruido que el rumor estridente de los remolinos de polvo, trapos, pajas y papeles que iban de arroyo en arroyo, de acera en acera, de esquina en esquina revolando y persiguiéndose, como mariposas que se buscan y huyen y que el aire envuelve en sus pliegues invisibles. Cual turbas de pilluelos, aquellas migajas de la basura, aquellas sobras de todo se juntaban en un montón, parábanse como dormidas un momento y brincaban de nuevo sobresaltadas, dispersándose, trepando unas por las paredes hasta los cristales temblorosos de los faroles, otras hasta los carteles de papel mal pegado a las esquinas, y había pluma que llegaba a un tercer piso, y arenilla que se incrustaba para días, o para años, en la vidriera de un escaparate, agarrada a un plomo. Vetusta, la muy noble y leal ciudad, corte en lejano siglo, hacía la digestión del cocido y de la olla podrida, y descansaba oyendo entre sueños el monótono y familiar zumbido de la campana de coro, que retumbaba allá en lo alto de la esbeltatorre en la Santa Basílica. La torre de la catedral, poema romántico de piedra,delicado himno, de dulces líneas de belleza muda y perenne, era obra del siglo diez y seis, aunque antes comenzada, de estilo gótico, pero, cabe decir, moderado por uninstinto de prudencia y armonía que modificaba las vulgares exageraciones de estaarquitectura. La vista no se fatigaba contemplando horas y horas aquel índice depiedra que señalaba al cielo; no era una de esas torres cuya aguja se quiebra desutil, más flacas que esbeltas, amaneradas, como señoritas cursis que aprietandemasiado el corsé; era maciza sin perder nada de su espiritual grandeza, y hasta sussegundos corredores, elegante balaustrada, subía como fuerte castillo, lanzándosedesde allí en pirámide de ángulo gracioso, inimitable en sus medidas y proporciones.Como haz de músculos y nervios la piedra enroscándose en la piedra trepaba a la altura, haciendo equilibrios de acróbata en el aire; y como prodigio de juegosmalabares, en una punta de caliza se mantenía, cual imantada, una bola grande debronce dorado, y encima otra más pequenya, y sobre ésta una cruz de hierro que acababaen pararrayos.'
bits_o=12
bits_l=4
S=2**bits_o
L=2**bits_l

import time
start_time = time.clock()
mensaje_codificado=LZ77Code(mensaje,S,L)
print (time.clock() - start_time, "seconds code")
start_time = time.clock()
mensaje_recuperado=LZ77Decode(mensaje_codificado)
print (time.clock() - start_time, "seconds decode")
ratio_compresion=8*len(mensaje)/((bits_o+bits_l+8)*len(mensaje_codificado))
print('Longitud de mensaje codificado:', len(mensaje_codificado))
print('Ratio de compresión:', ratio_compresion)



