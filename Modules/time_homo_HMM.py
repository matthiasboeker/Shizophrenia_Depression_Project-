#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:05:17 2020

@author: matthiasboeker
Function for the application of time independent HMM
Apply the hmmlearn package
The function is executed in the corresponding Jupyter notebook for the analysis
"""
import os
os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/')
import numpy as np
from Modules.func.support_functions import *
#from Modules.func.help_functions import *
from hmmlearn import hmm

def load_in_HMM_models(th_models_p, th_models_c):

    shizophrenia_p, shizophrenia_c = load_data()
    #1.2 set external parameters
    N = 2
    #Reshape the ts
    shizophrenia_p = [np.array(X).reshape(len(X), 1) for X in shizophrenia_p]
    shizophrenia_c = [np.array(X).reshape(len(X), 1) for X in shizophrenia_c]

    models_p = [hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100) for l in range(0,len(shizophrenia_p))]
    models_c = [hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100) for k in range(0,len(shizophrenia_c))]


    #Fit the models with EM
    for l  in range(0,len(shizophrenia_p)):
        #print('Fit Patient model: ', l)
        th_models_p.append(models_p[l].fit(shizophrenia_p[l]))

    for k in range(0, len(shizophrenia_c)):
        #print('Fit Control model: ', k)
        th_models_c.append(models_c[k].fit(shizophrenia_c[k]))
