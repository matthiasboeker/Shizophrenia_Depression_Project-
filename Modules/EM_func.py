#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:12:38 2020

@author: matthiasboeker
Execute main for time dependent HMM
Main script runs the Baum Welch Algorithm for Hidden Markov Models
The covariate can be added optional within the script BW_func
1. Script loads in time series
2. Sets external parameters like iterations, number of states, threshold
3. Compute the covariate which is supposed to be added
4. Load in boundaries for the covariate integration - optimization problem
5. The hmmlearn package provides the model objects which are initialized
6. For every time series the Baum Welch Algorithm is conducted and the results are directly written in a csv files
"""

import os
os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/')
import numpy as np
from Modules.func.support_functions import *
from Modules.func.help_functions import *
from Modules.func.BW_func import *
from Modules.func.result_entities import *
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
    cycles = range(0,200)
    tol = 1e-3
    ind = 1

    #The modelled Covariate
    Z_p = trig_cov(shizophrenia_p, None)
    Z_c = trig_cov(shizophrenia_c, None)

    #Load boundaries for optimization
    bnds = load_boundaries(N,ind)

    #init models
    models_p = [hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100) for l in range(0,len(shizophrenia_p))]
    models_c = [hmm.GaussianHMM(n_components=N,  covariance_type="diag", random_state=0,n_iter=100) for k in range(0,len(shizophrenia_c))]

    #Fit the models with EM
    for l  in range(0,len(shizophrenia_p)):
        print('Fit Patient model: ', l)
        models_p[l]._init(shizophrenia_p[l])
        #Ensure that resting state on place zero --> consistant transition matrix
        maxi = np.max(models_p[l].means_)
        mini = np.min(models_p[l].means_)
        models_p[l].means_[1] = maxi
        models_p[l].means_[0] = mini
        EA_func(models_p[l], N, shizophrenia_p[l], Z_p[l], cycles, bnds, ind, tol)
        save_res("_patient_%s" %l, models_p[l])


    for k in range(0, len(shizophrenia_c)):
        print('Fit Control model: ', k)
        models_c[k]._init(shizophrenia_c[k])

        #Ensure that resting state on place zero --> consistant transition matrix
        maxi = np.max(models_c[k].means_)
        mini = np.min(models_c[k].means_)
        models_c[k].means_[1] = maxi
        models_c[k].means_[0] = mini

        EA_func(models_c[k], N, shizophrenia_c[k], Z_c[k], cycles, bnds, ind, tol)
        save_res("_control_%s" %k, models_c[k])


if __name__ == "__main__":
    main()
