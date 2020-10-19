#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:12:38 2020

@author: matthiasboeker
Execute main time-inhomogenous HMM"""

import os
os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/')
import numpy as np 
from Modules.func.support_functions import *
from Modules.func.help_functions import *
from Modules.func.BW_func import *
from Modules.Covariate_model import trig_cov
from hmmlearn import hmm
from hmmlearn import _hmmc


def main():
    
    
    shizophrenia_p, shizophrenia_c = load_data()
    #Reshape 
    shizophrenia_p = [np.array(X).reshape(len(X), 1) for X in shizophrenia_p]
    shizophrenia_c = [np.array(X).reshape(len(X), 1) for X in shizophrenia_c]
    
    #Set external parameters 
    N = 2
    cycles = range(0,5) 
    tol = 1e-5
    ind = 1
    
    #The modelled Covariate 
    Z_p = trig_cov(shizophrenia_p, None)
    Z_c = trig_cov(shizophrenia_c, None)
    
    #Load boundaries for optimization    
    bnds = load_boundaries(N,ind)
    
    #init models 
    #models_p = [hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100) for l in range(0,len(shizophrenia_p))]
    models_c = [hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100) for k in range(0,len(shizophrenia_c))]
    
    #Fit the models with EM
    #for l  in range(1,len(shizophrenia_p)):
    #    print('Fit Patient model: ', l)
    #    models_p[l]._init(shizophrenia_p[l])
    #    EA_func(models_p[l], N, shizophrenia_p[l], Z_p[l], cycles, bnds, ind, tol)        
    #    save_res("cov_patient_%s" %l, models_p[l])
    
    for k in range(0, len(shizophrenia_c)):
        print('Fit Control model: ', k)
        models_c[k]._init(shizophrenia_c[k])
        EA_func(models_c[k], N, shizophrenia_c[k], Z_c[k], cycles, bnds, ind, tol) 
        save_res("cov_control_%s" %k, models_c[k])
        

if __name__ == "__main__":
    main()
