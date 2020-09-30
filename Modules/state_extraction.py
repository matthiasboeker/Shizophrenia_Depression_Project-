#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:10:25 2020

@author: matthiasboeker
Conduct the Bayesian Filtered GMM clustering for the Awake and Sleep State
"""
import sklearn.mixture as mix
from matplotlib import cm
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project')
from Modules.support_functions import *
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
patients_info = pd.read_csv('patients_info.csv')
#Import demographics on control group 
control_info = pd.read_csv('scores.csv')
#Import days 
days = pd.read_csv('days.csv')
shizophrenia_p, shizophrenia_c = preprocess(days,shizophrenia_p, shizophrenia_c)


#Apply the Gaussian Filter with sigma 10 according to Signal Processing.py outcome 
series_p = range(0,len(shizophrenia_p))
filtered_p = [gaussian_filter(shizophrenia_p[n], sigma=60) for n in series_p]
series_c = range(0,len(shizophrenia_c))
filtered_c = [gaussian_filter(shizophrenia_c[n], sigma=60) for n in series_c]


#Gaussian Mixture Model with four states 
model_f =  mix.GaussianMixture(4, covariance_type='full', random_state=0)

#Patients
model_states_p = [model_f.fit(np.array(patient).reshape(-1,1)).predict(np.array(patient).reshape(-1,1)) for patient in filtered_p ]
model_prob_p = [model_f.fit(np.array(patient).reshape(-1,1)).predict_proba(np.array(patient).reshape(-1,1)) for patient in filtered_p ]
model_means_p = [model_f.fit(np.array(patient).reshape(-1,1)).means_ for patient in filtered_p ]
model_cov_p = [model_f.fit(np.array(patient).reshape(-1,1)).covariances_ for patient in filtered_p ]


#Control 
model_states_c = [model_f.fit(np.array(control).reshape(-1,1)).predict(np.array(control).reshape(-1,1)) for control in filtered_c ]
model_prob_c = [model_f.fit(np.array(control).reshape(-1,1)).predict_proba(np.array(control).reshape(-1,1)) for control in filtered_c ]
model_means_c = [model_f.fit(np.array(control).reshape(-1,1)).means_ for control in filtered_c ]
model_cov_c = [model_f.fit(np.array(control).reshape(-1,1)).covariances_ for control in filtered_c ]


#Generate single state snippets of only sleep awake
#Find state number which represents the sleeping periods 
sleep_state_p = [np.argmin(model_means_p[n]) for n in series_p]
sleep_state_c = [np.argmin(model_means_c[n]) for n in series_c]

#Differenciate in resting & active/ distinguish
wake_snippets_p = [model_states_p[n] != sleep_state_p[n] for n in series_p]
wake_snippets_c = [model_states_c[n] != sleep_state_c[n] for n in series_c]

sleep_snippets_p = [model_states_p[n] != sleep_state_p[n] for n in series_p]
sleep_snippets_c = [model_states_c[n] != sleep_state_c[n] for n in series_c]




binn_p = [shizophrenia_p[n].loc[wake_snippets_p[n]] for n in series_p]
binn_c = [shizophrenia_c[n].loc[wake_snippets_c[n]] for n in series_c]

#Extract the Wake periods 
patients_wake_periods = [snippets(k) for k in binn_p]
control_wake_periods = [snippets(k) for k in binn_c]

n = 10000
for i in range(1,15):
    plt.scatter(gaussian_filter(shizophrenia_c[i+1], sigma=8)[:n-10], gaussian_filter(shizophrenia_c[i+1], sigma=8)[10:n], marker = '.',linewidths = 0.1, label = 'Control')
    plt.scatter(gaussian_filter(shizophrenia_p[i], sigma=8)[:n-10], gaussian_filter(shizophrenia_p[i], sigma=8)[10:n], marker = '.',linewidths = 0.1, label = 'Patient')
    plt.legend()
    plt.show()