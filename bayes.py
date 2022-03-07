#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 22:28:25 2022

@author: miguel
"""
 # 
 #  BAYES 
 #
 #  EJEMPLO DE USO:
 #    
 #    clasificador = N_Bayes()
 #    clasificador.load_data(train_attrib, train_labels)
 #    res = clasificador.bayes(atrib)
 
 #  PARAMETROS:
                
 #  Arreglo de atributos ([[float]])
             
 #  train_attrib = [[Atributo(1 ,1);   Atributo(1 ,2);   Atributo(1 ,..);   Atributo(1 ,N)],
 #                  [Atributo(2 ,1);   Atributo(2 ,2);   Atributo(2 ,..);   Atributo(2 ,N)],
 #                  [Atributo(... ,1); Atributo(... ,2); Atributo(... ,..); Atributo(... ,N)],
 #                  [Atributo(N,1);    Atributo(N,2);    Atributo(N ,..);   Atributo(N,N)]]
 #  Donde los indices del arreglo corresponden a (# de Atributo, indice de ejemplo)
 
 #  Arreglo de etiquetas
 #                   cadena de texto 
 #                   con numero entero        cadena de texto
 #                         ||                       ||
 #                         \/                       \/
 #
 #  train_labels = [[indice de ejemplo1,   etiqueta del ejemplo1]
 #                  [indice de ejemplo2,   etiqueta del ejemplo2],
 #                  [indice de ejemplo..., etiqueta del ejemplo...]
 #                  [indice de ejemploN,   etiqueta del ejemploN]]
 #
 #  Vector de atributos del ejemplo a clasificar
 
 #  atrib = [A1,A2,..,AN]
 
 #  SALIDA:
     
 #  [ETIQUETA, WEIGHT]
 
import math
import statistics
import numpy as np

class N_Bayes:
    
    def __init__(self):
        self.data = []
    
    #P(A|C)
    def __gauss(self, prom, dev, val):
        b = (math.pow((val-prom),2)/(2*math.pow(dev,2)))*-1
        return (1/(math.sqrt(2*math.pi)*dev))*math.exp(b)
    
    def bayes(self,atributos):
        resultado = ["-",0.0]
        for clase in self.data:
            nombre = clase[0]
            p_clase=clase[1]  #P(C)
            #P(A1,A2,A3,A4|C) :
            p_a = 1.0
            #print(clase)
            for a_i in range(len(clase[2])):
                #print(clase[2][a_i],clase[3][a_i],atributos[a_i])
                p_a = p_a * self.__gauss(clase[2][a_i],clase[3][a_i],atributos[a_i])
                #print(p_a)
            #P(A1,A2,A3,A4|C)*P(C)
            res = (p_a*p_clase)
            #print(nombre, res)
            if (res > resultado[1]):
                resultado = [nombre, res]
        
        #print(resultado)
        return resultado
    
    def bayes_log(self,atributos):
        resultado = ["-",0.0]
        for clase in self.data:
            nombre = clase[0]
            p_clase=clase[1]  #P(C)
            #P(A1,A2,A3,A4|C) :
            p_a = 0.0
            #print(clase)
            for a_i in range(len(clase[2])):
                #print(clase[2][a_i],clase[3][a_i],atributos[a_i])
                p_a = p_a + math.log(self.__gauss(clase[2][a_i],clase[3][a_i],atributos[a_i]))
                #print(p_a)
            #P(A1,A2,A3,A4|C)*P(C)
            res = (p_a*p_clase)
            #print(nombre, res)
            if (res < resultado[1]):
                resultado = [nombre, res]
        
        #print(resultado)
        return resultado
        
    
    def load_data(self, attrib,labels):
        clases = [[],[]]
        buf=[]
        poblacion = len(labels)
        labels = sorted(labels, key=lambda c : c[1])
        #print(attrib, labels)
        for d in labels:
            nombre = d[1]
            attrib_index = int(d[0])
            if (nombre not in clases[0]):
                clases[0].append(nombre)
                for a in attrib:
                    buf.append([a[attrib_index]])
                clases[1].append(buf.copy())
                buf.clear()
            else:
                i = clases[0].index(nombre)
                for i_a in range(len(attrib)):
                    clases[1][i][i_a].append(attrib[i_a][attrib_index])
        buf.clear()
        for i in range(len(clases[0])):
            nombre = clases[0][i]
            prob_total = len(clases[1][i][0])/poblacion
            prom_attrib = []
            devstd_attrib = []
            for a in clases[1][i]:
                x = statistics.mean(a)
                s = statistics.stdev(a)
                prom_attrib.append(x)
                devstd_attrib.append(s)
            buf.append([nombre,prob_total,prom_attrib,devstd_attrib])
        
        #print("[[clase, prob_total, prom_attrib, devstd_attrib]]")
        #print(buf)
        self.data = buf
            
                
                
        
        
        