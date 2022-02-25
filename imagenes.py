#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:37:16 2022

@author: miguel
"""

from PIL import Image
import numpy as np
import random
import os

def open_and_convert(file):
    # Abrir imagen
    imagen = Image.open(file)
    #print(type(imagen))
    #print("loading: ", file)
    
    # Imprimir atributos
    #print(imagen.mode)
    #print(imagen.size)
    W, H = imagen.size
    
    imagen = Image.open(file).resize((int(W/10),int(H/10)))
    #imagen = Image.open(file).resize((32,32))
    #imagen = Image.open(file).resize((16,16))
    imagen = imagen.convert('L')
    #imagen.show()
    
    # Convertir a array
    img_np = np.array(imagen)
    img_h, img_w = img_np.shape
    #print("Array shape:", img_h, img_w)
    
    # Reshape a 1-D
    tam = img_h*img_w
    linear = img_np.reshape(1,tam)
    #linear.dtype = np.float32
    #print("Shape final: ", linear.shape)
    #print(linear[0].tolist())
    
    # Normalizar
    #n = linear/np.linalg.norm(linear)
    return linear[0].tolist()

def add_to_array(attrib,data):
    if (len(attrib) == 0):
        for i in data:
            attrib.append([float(i)])
    else:
        for i in range(len(data)):
            attrib[i].append(float(data[i]))
   
def load_data(dataset_path, n_muestras_train = 25):
    attrib_e = []
    labels_e = []
    
    attrib_p = []
    labels_p = []
    
    file_names = os.listdir(dataset_path)
    
    #Barajeo
    random.shuffle(file_names)
        
    for i in range(len(file_names)):
        if (i < n_muestras_train):
            #print("entrenamiento ", i, buffer[i])
            add_to_array(attrib_e, open_and_convert(dataset_path + file_names[i]))
            if "m" in file_names[i]:
                label = "mujer"
            else:
                label = "hombre"
            labels_e.append((i,label))
        else:
            #print("pruebas", i, buffer[i])
            add_to_array(attrib_p, open_and_convert(dataset_path + file_names[i]))
            if "m" in file_names[i]:
                label = "mujer"
            else:
                label = "hombre"
            labels_p.append((i-n_muestras_train,label))
            
    return attrib_e, labels_e, attrib_p, labels_p
        



