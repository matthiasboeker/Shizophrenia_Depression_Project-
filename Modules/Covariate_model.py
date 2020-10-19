#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:39:45 2020

@author: matthiasboeker
Modelling the covariate 
"""
import numpy as np

def trig_cov(shizophrenia, max_len = None):
    return [np.array(np.cos(((len(X[:max_len]))/60/24)*2*np.pi*np.arange(0,len(X[:max_len]))/len(X[:max_len]))+1) for X in shizophrenia]
    
