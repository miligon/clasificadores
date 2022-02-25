#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:02:45 2022

@author: miguel
"""
import csv_file
import random
from bayes import N_Bayes
from knn import KNN
import imagenes
import statistics

def ejecuta_clasificador_bayes_iris(n_training):

    
    file_name = "data/iris.data"
    n_testing = 150-n_training
    print("# datos de entrenamiento:", n_training, ", # de datos prueba:", n_testing)
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels = csv_file.load_data(file_name, n_training)
    
    clasificador = N_Bayes()
    #Calcula medias, desviacion estandar, etc.
    #print(train_attrib)
    clasificador.load_data(train_attrib, train_labels)
    
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib[0])):
        a1 = test_attrib[0][a]
        a2 = test_attrib[1][a]
        a3 = test_attrib[2][a]
        a4 = test_attrib[3][a]
        
        atrib = [a1,a2,a3,a4]
        #print("prueba: ", a, " Data:", atrib)
        
        res = clasificador.bayes(atrib)
        
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res, "OK")
        else:
            bad = bad +1
            #print("Esperado:", name, " Obtenido:", res, "FALLÓ")
            
    good_p = (good/n_testing)*100
    bad_p = (bad/n_testing)*100
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]

def ejecuta_clasificador_bayes_fei(n_training):

    
    dataset_path = "fei/frontalimages_manuallyaligned_part1/"
    n_testing = 200-n_training
    print("# datos de entrenamiento:", n_training, ", # de datos prueba:", n_testing)
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels = imagenes.load_data(dataset_path, n_training)
    
    clasificador = N_Bayes()
    #Calcula medias, desviacion estandar, etc.
    clasificador.load_data(train_attrib, train_labels)
    
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib[0])):
        atrib = []
        for i in range(len(test_attrib)):
            atrib.append(test_attrib[i][a])
        #print("prueba: ", a, " Data:", atrib)
        
        res = clasificador.bayes_log(atrib)
        
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res, "OK")
        else:
            bad = bad +1
            #print("Esperado:", name, " Obtenido:", res, "FALLÓ")
            
    good_p = (good/n_testing)*100
    bad_p = (bad/n_testing)*100
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]

def ejecuta_clasificador_knn_fei(n_training):

    
    dataset_path = "fei/frontalimages_manuallyaligned_part1/"
    n_testing = 200-n_training
    print("# datos de entrenamiento:", n_training, ", # de datos prueba:", n_testing)
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels = imagenes.load_data(dataset_path, n_training)
    
    clasificador = KNN()
    #Calcula medias, desviacion estandar, etc.
    clasificador.load_data(train_attrib, train_labels)
    
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib[0])):
        atrib = []
        for i in range(len(test_attrib)):
            atrib.append(test_attrib[i][a])
        #print("prueba: ", a, " Data:", atrib)
        
        res = clasificador.knn_classify(atrib)
        
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res, "OK")
        else:
            bad = bad +1
            #print("Esperado:", name, " Obtenido:", res, "FALLÓ")
            
    good_p = (good/n_testing)*100
    bad_p = (bad/n_testing)*100
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]

def ejecuta_clasificador_knn_iris(n_training):
    
    file_name = "data/iris.data"
    n_testing = 150-n_training
    print("# datos de entrenamiento:", n_training, ", # de datos prueba:", n_testing)
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels = csv_file.load_data(file_name, n_training)
    
    clasificador = KNN()
    
    clasificador.load_data(train_attrib, train_labels)
    
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib[0])):
        a1 = test_attrib[0][a]
        a2 = test_attrib[1][a]
        a3 = test_attrib[2][a]
        a4 = test_attrib[3][a]
        
        atrib = [a1,a2,a3,a4]
        #print("prueba: ", a, " Data:", atrib)
        
        res = clasificador.knn_classify(atrib)
        print(res)
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res, "OK")
        else:
            bad = bad +1
            #print("Esperado:", name, " Obtenido:", res, "FALLÓ")
            
    good_p = (good/n_testing)*100
    bad_p = (bad/n_testing)*100
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]


if __name__ == '__main__':
    n_iter = 10
    e_good = []
    e_bad = []
    for i in range(n_iter):
        n_test = 50
        
        # Seleccion de prueba a ejecutar
        good, bad = ejecuta_clasificador_knn_fei(n_test)
        #good, bad = ejecuta_clasificador_knn_iris(n_test)
        #good, bad = ejecuta_clasificador_bayes_iris(n_test)
        #good, bad = ejecuta_clasificador_bayes_fei(n_test)
        e_good.append(good)
        e_bad.append(bad)
    
    prom_e_good = round(statistics.mean(e_good))
    prom_e_bad = round(statistics.mean(e_bad))
    
    dev_std_good = statistics.stdev(e_good)
    dev_std_bad =  statistics.stdev(e_bad)
    
    print("promedio Good: ", prom_e_good, "% , promedio bad: ", prom_e_bad, "%, # de iteraciones: ", n_iter)
    print("dev std: ", dev_std_good, "% , promedio bad: ", dev_std_bad, "%")