#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:01:50 2022

@author: miguel
"""

import numpy as np
import math

class KNN:
    def __init__(self):
        self.data = []
        
    def load_data(self, attrib,labels):
        self.data = attrib
        self.labels = labels
        
    def get_distance(self,origen,destino):
        a = np.array(origen)
        b = np.array(destino)
        c = b-a
        c = np.power(c, 2)
        total = np.sum(c)
        return math.sqrt(total)
    
    def extract_attributes(self,index):
        res = []
        for a in self.data:
            res.append(a[index])
        return res
    
    def calc_peso(self, distance):
        # En caso de que el test y el punto a comparar sean identicos
        return math.exp((-1*math.pow(distance,2))/(2*math.pow(self.sigma,2)))
            
    
    def get_vecinos(self,t_data, num):
        test = np.array(t_data)
        distancias = []
        for i in range(len(self.data[0])):
            label = self.labels[i][1]
            distancia = self.get_distance(test, self.extract_attributes(i))
            if distancia == 0:
                return [[distancia, label]]
            distancias.append([distancia, label])
        # orden ascendente
        distancias.sort(reverse=False, key=lambda dist: dist[0])
        self.dmin = distancias[0][0]
        self.sigma = 2*self.dmin
        return distancias[0:num]
    
    def simple_knn(self, vecinos):
        conteo_vecinos = []
        for valores in vecinos:
            dist = valores[0]
            conteo_vecinos.append(valores[1])
        voto = max(conteo_vecinos, key=conteo_vecinos.count)
        return voto
    
    def weighted_predict(self, vecinos):
        vecinos.sort(key=lambda label: label[1])
        votos = []
        labels = []
        for vecino in vecinos:
            label = vecino[1]
            distancia = vecino[0]
            weight = self.calc_peso(distancia)
            if label not in labels:
                votos.append(weight)
                labels.append(label)
            else:
                index = labels.index(label)
                votos[index] = votos[index] + weight       
        #print(labels, votos)
        v_mayor = 0.0
        l_mayor = ""
        for i in range(len(labels)):
            if (votos[i] > v_mayor):
                v_mayor = votos[i]
                l_mayor = labels[i]
        
        return [l_mayor, v_mayor]
            
    
    def knn_classify(self,atrib, k):
        vecinos = self.get_vecinos(atrib, k)
        if (len(vecinos) == 1):
            print("(muestra==test, d=0): ", vecinos[0][1])
            return [vecinos[0][1],100]
        res = self.weighted_predict(vecinos)
        #print("Predicción: ", res)
        return res
        
        