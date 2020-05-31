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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



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
    
