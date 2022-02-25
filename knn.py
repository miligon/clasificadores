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
            
    
    def get_vecinos(self,t_data, num):
        test = np.array(t_data)
        distancias = []
        for i in range(len(self.data[0])):
            label = self.labels[i][1]
            distancias.append([self.get_distance(test, self.extract_attributes(i)), label])
        distancias.sort(key=lambda dist: dist[0])
        return distancias[0:num]
    
    def knn_classify(self,atrib):
        vecinos = self.get_vecinos(atrib, 4)
        #print(vecinos)
        conteo_vecinos = []
        for valores in vecinos:
            dist = valores[0]
            conteo_vecinos.append(valores[1])
        voto = max(conteo_vecinos, key=conteo_vecinos.count)
        #print(voto)        
        return [voto, 1.0]
        
        