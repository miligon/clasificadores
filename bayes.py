#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 22:28:25 2022

@author: miguel
"""
import math
import statistics

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
            attrib_index = d[0]
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
            
                
                
        
        
        