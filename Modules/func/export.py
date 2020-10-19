#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:16:15 2020

@author: matthiasboeker
Export the time series data for matlab and R
"""
import numpy as np
import os


os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/')
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


os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data')
import csv
from itertools import zip_longest

with open("patients_ts.csv","w+") as f:
    writer = csv.writer(f)
    for values in zip_longest(*shizophrenia_p):
        writer.writerow(values)

os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data')
import csv
from itertools import zip_longest

with open("control_ts.csv","w+") as f:
    writer = csv.writer(f)
    for values in zip_longest(*shizophrenia_c):
        writer.writerow(values)