#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:05:17 2020

@author: matthiasboeker
Execute main time-inhomogenous HMM"""
import os
os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/')
import numpy as np 
from Modules.func.support_functions import *
from Modules.func.help_functions import *
from hmmlearn import hmm
    
def main():

    shizophrenia_p, shizophrenia_c = load_data()
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



        
if __name__ == "__main__":
    main()