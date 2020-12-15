#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:18:57 2020

@author: matthiasboeker
Apply the viterbi algo 
"""
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt

os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project')
from Modules.func.result_entities import *
from Modules.func.BW_func import *
from Modules.func.load_vitberi import *
from Modules.func.support_functions import *

patients_res, control_res = load_entities(directory='/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Results/latest_run_03.11.with_C/')
shizophrenia_p, shizophrenia_c = load_data()
shizophrenia_p = [np.array(X).reshape(len(X), 1) for X in shizophrenia_p]
shizophrenia_c = [np.array(X).reshape(len(X), 1) for X in shizophrenia_c]


state_seq_p = [viterbi(len(shizophrenia_p[i]), 2 ,shizophrenia_p[i] ,np.log(patients_res[i].start_prob),
             np.log(patients_res[i].trans_mat), patients_res[i].means,  patients_res[i].cov) for i in range(0,len(shizophrenia_p))]
state_seq_c = [viterbi(len(shizophrenia_c[i]), 2 ,shizophrenia_c[i] ,np.log(control_res[i].start_prob),
             np.log(control_res[i].trans_mat), control_res[i].means,  control_res[i].cov) for i in range(0,len(shizophrenia_c))]


load_state_sequence(shizophrenia_p[1],patients_res[1] )

patients_res[1].state_seq





def load_state_sequence(X, entity):
    
    entitiy.state_seq = viterbi(len(X[i]), entity.components ,X ,np.log(entity.start_prob),
             np.log(entity.trans_mat), entity.means,  entity.cov)
