#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:15:06 2020

@author: matthiasboeker
Several supportive functions applied throughout the script
1. Function to save the results in csv
2. Funtion to load in the data
3. Preprocessing functions
4. etc. 
"""
import re
import os
import pandas as pd
import datetime as dt
import numpy as np
from statsmodels.tsa.stattools import kpss, adfuller

def save_res(prefix, base):
    os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Results')
    trans = np.vstack(base.transmat_).T
    cov = base.covars_.reshape((2,1))
    mean = base.means_.reshape((2,1))
    start_prob = base.startprob_.reshape((2,1))
    log_prob = base.log_prob

    coef = np.hstack(base.link_coef)
    res = np.concatenate((cov,mean,start_prob, coef), axis = 1)
    name_t = "%s_Transition_matrix.csv" % prefix
    name_r = "%s_Results.csv" % prefix
    name_l = "%s_log_prob.csv" % prefix
    np.savetxt(name_t, trans, delimiter=",")
    np.savetxt(name_r, res, delimiter=",", header= "covariance, means, start_prob, link_coef01, link_coef02, link_coef11, link_coef12", comments='' )
    np.savetxt(name_l, log_prob, delimiter=",")





def load_data():

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
    return shizophrenia_p, shizophrenia_c



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def kpss_test(timeseries):
    #print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    #print (kpss_output)
    return kpss_output[1]

def adf_test(timeseries):
    #print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    return dfoutput[1]


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

    ind = [0,1,2,3,4,5,6,30,31]
    for o in ind:
        shizophrenia_c[o] = shizophrenia_c[o][1*18*60:]
        shizophrenia_c[o] = shizophrenia_c[o].reset_index()


    shizophrenia_c[26] = shizophrenia_c[26][50:].reset_index()
    shizophrenia_c[27] = shizophrenia_c[27][1*23*60:].reset_index()
    shizophrenia_c[28] = shizophrenia_c[28][1*23*60:].reset_index()
    shizophrenia_c[29] = shizophrenia_c[29][(1*21*60)-30:].reset_index()

    #Cut the Signals to the amount of days documented
    for k in range(0,len(shizophrenia_p)):
        shizophrenia_p[k] = shizophrenia_p[k]['activity'][:days_p['length'][k]]
    for l in range(0,len(shizophrenia_c)):
        shizophrenia_c[l] = shizophrenia_c[l]['activity'][:days_c['length'][l]]
    return(shizophrenia_p, shizophrenia_c)



def calc_timeshift(data_start,shift_start):
    date = dt.date(1, 1, 1)
    start = dt.datetime.combine(date, data_start)
    shift = dt.datetime.combine(date, shift_start)
    if (start>shift):
        #data_start 15:00
        #shift_start 09:00
        diff = shift - start
    else:
        #data_start 09:00
        #shift star
        diff = shift - start
    return(int(diff.seconds/3600))

def shift_cols(data, diff):
    data = data.iloc[diff*60:,:]
    return(data)


def get_intervals(data, intervals = 0):
    if intervals == 0:
        binn = list()
        ac_start = data['timestamp'].iloc[0]
        #starting daily intervals at 12am -12pm
        time = dt.time(12, 0, 0)
        start = '2018-04-24 12:00:00'
        end = '2018-04-25 11:59:00'
        diff = calc_timeshift(ac_start.time(),time)
        data = shift_cols(data,diff)
        data = data.drop(['timestamp'], axis=1)
        for g, df in data.groupby(np.arange(len(data)) // (60*24)):
            df.index = pd.date_range(start=start, periods=len(df), freq='min')
            binn.append(df)
        datafr = pd.concat(binn,axis=1, ignore_index= True)
        return datafr
    if intervals == 1:
        bin_d = list()
        ac_start = data['timestamp'].iloc[0]
        #starting daily intervals at 9am -9pm
        time = dt.time(9, 0, 0)
        start_d = '2018-04-24 09:00:00'
        diff = calc_timeshift(ac_start.time(),time)
        data = shift_cols(data,diff)
        data = data.drop(['timestamp'], axis=1)
        for g, df in data.groupby(np.arange(len(data)) // (60*12)):
            if ((g+2)%2)==0:
                df.index = pd.date_range(start=start_d, periods=len(df), freq='min')
                bin_d.append(df)
        data_d = pd.concat(bin_d,axis=1, ignore_index= True)
        return data_d

    if intervals == 2:
        bin_n = list()
        ac_start = data['timestamp'].iloc[0]
        #starting daily intervals at 9am -9pm
        time = dt.time(9, 0, 0)
        start_n = '2018-04-24 20:59:00'
        diff = calc_timeshift(ac_start.time(),time)
        data = shift_cols(data,diff)
        data = data.drop(['timestamp'], axis=1)
        for g, df in data.groupby(np.arange(len(data)) // (60*12)):
            if ((g+2)%2)!=0:
                df.index = pd.date_range(start=start_n, periods=len(df), freq='min')
                bin_n.append(df)
        data_n = pd.concat(bin_n,axis=1, ignore_index= True)
        return data_n

def confusion_matrix(y,y_hat):
    confusion_matrix = np.zeros((2,2))
    #True Positive
    confusion_matrix[0,0] = np.sum(np.logical_and((y == 1), (y_hat==1))*1)
    #True Negative
    confusion_matrix[1,1] = np.sum(np.logical_and((y == 0), (y_hat==0))*1)
    #False positive
    confusion_matrix[0,1] = np.sum(np.logical_and((y == 1), (y_hat==0))*1)
    #False negative
    confusion_matrix[1,0] = np.sum(np.logical_and((y == 0), (y_hat==1))*1)
    return confusion_matrix

def binary_classifier(confusion_matrix, score = 'a'):

    if score == 'a':
        accuracy = (confusion_matrix[1,1]+confusion_matrix[0,0])/(np.sum(confusion_matrix))
        return(accuracy)
    if score == 'r':
        recall = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
        return recall
    if score == 'fpr':
        fpr = confusion_matrix[0,1]/(confusion_matrix[0,1]+confusion_matrix[1,1])
        return fpr
    if score == 'fone':
        f1 = (2*confusion_matrix[0,0])/(2*confusion_matrix[0,0]+confusion_matrix[0,1]+confusion_matrix[1,0])
        return f1
    if score == 'mcc':
         mcc = ((confusion_matrix[1,1]*confusion_matrix[0,0])-(confusion_matrix[0,1]*confusion_matrix[1,0]))/np.roots((confusion_matrix[0,0]+confusion_matrix[0,1])*(confusion_matrix[0,0]+confusion_matrix[1,0])*(confusion_matrix[1,1]+confusion_matrix[0,1])*(confusion_matrix[1,1]+ confusion_matrix[1,0]))
         return mcc
    else:
        print('Choose score: a, r, fpr, fone, mmcc')


def snippets(k):
    breaker = k.index[0]
    chunks = list()
    for l in range(0,len(k.index)-1):
        if k.index[l+1] > k.index[l]+1:
            chunks.append(k.loc[breaker:k.index[l]])
            breaker = k.index[l+1]
    return(chunks)
