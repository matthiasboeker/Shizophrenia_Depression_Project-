#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:25:47 2020

@author: matthiasboeker
Read in results and create objects 
"""
import numpy as np
import pandas as pd
import os
from Modules.func.support_functions import *


class result_en():
    
    def __init__(self, name, group, variant,samples = 1, components = 2, trans_mat = 0, start_prob = 0, 
                 means = 0, cov=1.0, link1=1.0, link2=1.0 , log_prob = 0):
        self.name = name 
        self.group = group
        self.variant = variant 
        self.trans_mat = trans_mat
        self.start_prob = start_prob
        self.means = means 
        self.cov = cov 
        self.link1 = link1
        self.link2 = link2 
        self.samples = samples
        self.components = components


             
def load_entities(directory='/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Results/With_covariate/'): 
    control_ent_list = []
    patient_ent_list = [] 
    os.chdir(directory+'control')
    files = os.listdir()
    files.sort(key=natural_keys)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    c=1
    for i in range(0,len(files),2):
        name = 'cov_control_%s'%c
        con = result_en(name, group = 'control', variant = 'with_cov')
        c=c+1
        binn = np.genfromtxt(files[i+1], delimiter = ',').T
        con.trans_mat = binn.reshape(2,2,binn.shape[1])
        con.samples = binn.shape[1]
        
        res = pd.read_csv(files[i], delimiter = ',')
        con.cov = res['covariance'].values
        con.start_prob = res[' start_prob'].values
        con.means = res[' means'].values
        con.link1 = pd.DataFrame([res[' link_coef01'],res[' link_coef02']])
        con.link2 = pd.DataFrame([res[' link_coef11'],res[' link_coef12']])
        control_ent_list.append(con)
     
        
    os.chdir(directory+'patient')
    files = os.listdir()
    files.sort(key=natural_keys)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    
    p=1
    for i in range(0,len(files),2):
        name = 'cov_patient_%s'%p
        pat = result_en(name, group = 'control', variant = 'with_cov')
        p=p+1
        
        binn = np.genfromtxt(files[i+1], delimiter = ',').T
        pat.trans_mat = binn.reshape(2,2,binn.shape[1])
        
        res = pd.read_csv(files[i], delimiter = ',')
        pat.cov = res['covariance'].values
        pat.start_prob = res[' start_prob'].values
        pat.means = res[' means'].values
        pat.link1 = pd.DataFrame([res[' link_coef01'],res[' link_coef02']])
        pat.link2 = pd.DataFrame([res[' link_coef11'],res[' link_coef12']])
        patient_ent_list.append(pat)

    return patient_ent_list, control_ent_list