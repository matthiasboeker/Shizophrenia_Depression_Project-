#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:12:38 2020

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
    cycles = range(0,10) 
    tol = 1e-5
    
    #1.3 Initial Covariates Xt
    #One covariate 
    ind = 1
    #The modelled Covariate 
    Z_p = [np.array(np.cos(((len(X))/60/24)*2*np.pi*np.arange(0,len(X))/len(X))+1) for X in shizophrenia_p]
    Z_c = [np.array(np.cos(((len(X))/60/24)*2*np.pi*np.arange(0,len(X))/len(X))+1) for X in shizophrenia_c]
    
    #Introduce boundaries for the optimization, set diag zero
    lb = -np.inf*np.ones([N,N,ind+1])
    [np.fill_diagonal(lb[:,:,l],0) for l in range(0,lb.shape[2])]
    ub = np.inf*np.ones([N,N,ind+1])
    [np.fill_diagonal(ub[:,:,l],0) for l in range(0,ub.shape[2])]
    bnds = optimize.Bounds(lb.flatten(),ub.flatten())
            
    
    model = hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100)

    #Fit the models with EM
    for l  in range(0,len(shizophrenia_p)):
        print('Fit Patient model: ', l)
        model._init(shizophrenia_p[l])
        models_p[l] = EA_func(model, N, shizophrenia_p[l], Z_p[l], tol, cycles, bnds, ind)
    
    for k in range(0, len(shizophrenia_c)):
        print('Fit Patient model: ', k)
        model._init(shizophrenia_c[k])
        models_c[k] = EA_func(model, N, shizophrenia_c[k], Z_c[k], tol, cycles, bnds, ind) 
        
