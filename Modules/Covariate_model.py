#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:39:45 2020

@author: matthiasboeker
Modelling the covariate:
The covariate is modelled as a trigonometric:
The functions takes in the list of time series and creates
the same number of covariate with the according length
"""
import numpy as np

def trig_cov(shizophrenia, max_len = None):
    return [np.array(np.cos(((len(X[:max_len]))/60/24)*2*np.pi*np.arange(0,len(X[:max_len]))/len(X[:max_len])))
            + np.array(np.sin(((len(X[:max_len]))/60/24)*2*np.pi*np.arange(0,len(X[:max_len]))/len(X[:max_len]))) for X in shizophrenia]
