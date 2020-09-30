#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:24:11 2020

@author: matthiasboeker
Activity signal preprocessing 
"""

import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt


def preprocess(days,shizophrenia_p,shizophrenia_c):    
    #Calculate the length of the signal
    days['length'] = 24*60*days['days'].astype(int)
    #Split the id to number and patient info to numeric 
    new = days['id'].str.split('_', n=1, expand=True)
    days['id'] = new[1].astype(int)
    days['type'] = new[0]
    #Sort the and split the dataframe by id
    days_p = days[days['type']=='patient'].sort_values(by=['id'])
    days_p= days_p.reset_index(drop=True)
    days_c = days[days['type']=='control'].sort_values(by=['id'])
    days_c= days_c.reset_index(drop=True)
    
    #Cut the Signals to the amount of days documented 
    for k in range(0,len(shizophrenia_p)):
        shizophrenia_p[k] = shizophrenia_p[k]['activity'][:days_p['length'][k]]
    for l in range(0,len(shizophrenia_c)):
        shizophrenia_c[l] = shizophrenia_c[l]['activity'][:days_c['length'][l]]
        
