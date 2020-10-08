#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:05:17 2020

@author: matthiasboeker
Execute main time-inhomogenous HMM"""
if __name__ == "__main__":
    import os
    os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/')
    import numpy as np 
    from Modules.support_functions import *
    from Modules.help_functions import *
    from Modules.BW_func import *
    from hmmlearn import hmm
    from hmmlearn import _hmmc
    
    #1. The initialization
    #1.1 load the data
    #Import Schizophrenia data
    os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data/psykose/patient')
    files = os.listdir()
    files.sort(key=natural_keys)
    shizophrenia_p = list()
    for i in range(0,len(files)):
        if files[i].endswith('.csv'):
            shizophrenia_p.append(pd.read_csv(files[i]))
        
    os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data/psykose/control')
    files = os.listdir()
    files.sort(key=natural_keys)
    shizophrenia_c = list()
    for i in range(0,len(files)):
        if files[i].endswith('.csv'):
            shizophrenia_c.append(pd.read_csv(files[i]))
    #Import demographics on Schizophrenia patients
    os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data/psykose')
    #Import days 
    days = pd.read_csv('days.csv')
    shizophrenia_p, shizophrenia_c = preprocess(days,shizophrenia_p, shizophrenia_c)

    #1.2 set external parameters 
    N = 2
    #Reshape the ts
    shizophrenia_p = [np.array(X).reshape(len(X), 1) for X in shizophrenia_p]
    shizophrenia_c = [np.array(X).reshape(len(X), 1) for X in shizophrenia_c]
    
    th_models_p = []
    th_models_c = []
    
    model = hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100)

    #Fit the models with EM
    for l  in range(0,len(shizophrenia_p)):
        print('Fit Patient model: ', l)
        th_models_p.append(model.fit(shizophrenia_p[l]))
    
    for k in range(0, len(shizophrenia_c)):
        print('Fit Control model: ', k)
        th_models_c.append(model.fit(shizophrenia_c[k]))
        
