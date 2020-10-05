#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:15:06 2020

@author: matthiasboeker
supportive function  
"""
import re
import pandas as pd
import datetime as dt 
import numpy as np
from statsmodels.tsa.stattools import kpss, adfuller 





def eval_embedded_dim(ts, tau = 100, maxnum = 500, dim = np.arange(8, 12)):
    try:
        f1, f2, f3 = dimension.fnn(np.array(ts), tau=tau, dim=dim, window=10, metric='euclidean', maxnum=maxnum)
    except:
        print('need higher dimensions')
    #return(np.argmin(f1))
    
    else:
        for kl in range(1,4):
            try:
                f1, f2, f3 = dimension.fnn(np.array(ts), tau=tau, dim=dim+kl, window=10, metric='euclidean', maxnum=maxnum)
            except: 
                print('fail in adding a dimension ',kl )
    return(np.argmin(f1))

    


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
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


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
        

def detect_circ(signal, adj = 0.03):
    from scipy.fft import fft, ifft
    from scipy.fft import fftfreq
    active = list()
    
    trans_fft = fft(signal)
    trans_psd = np.log10(np.abs(trans_fft) ** 2)
    freq = fftfreq(len(signal),(1. / 32))
    i = freq > 0
    max_peak = freq[np.argmax(trans_psd[i])]
    trans_fft_bis = trans_fft.copy()
    trans_fft_bis[np.abs(freq) > max_peak+adj] = 0
    back = pd.Series(np.real(ifft(trans_fft_bis)))
    deriv = np.gradient(back)

    asign = np.sign(deriv)
    signchanges = pd.Series(((np.roll(asign, 1) - asign) != 0).astype(int))
    extreme_val = signchanges.loc[signchanges==1].index
    deriv2 = np.gradient(deriv)
    asign2 = np.sign(deriv2)
    signchanges2 = pd.Series(((np.roll(asign2, 1) - asign2) != 0).astype(int))
    extreme_val2 = signchanges2.loc[signchanges2==1].index
    
    for i in range(0, len(extreme_val2)-1):
        ren_s = extreme_val2[i]
        ren_e = extreme_val2[i+1]
        active.append(signal[ren_s:ren_e])

            
    res = pd.DataFrame([], columns = ['Mean', 'Variance'])
    for i in range(0,len(active)):
        ins_ac = [np.mean(active[i]), np.var(active[i])]
        res = np.vstack([res, ins_ac])
        
    results = pd.DataFrame(res, columns = ['Mean', 'Variance'])
    
    return results, extreme_val2
   
def snippets(k):
    breaker = k.index[0]
    chunks = list()
    for l in range(0,len(k.index)-1):
        if k.index[l+1] > k.index[l]+1:
            chunks.append(k.loc[breaker:k.index[l]])
            breaker = k.index[l+1] 
    return(chunks)
    
            
            