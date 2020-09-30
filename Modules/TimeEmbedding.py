#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:15:00 2020

@author: matthiasboeker
#The time series embedding
#Literature
#Kantz H, Schreiber T. Nonlinear time series analysis. vol. 7. Cambridge university press; 2004
#Takens F. On the numerical determination of the dimension of an attractor. Springer; 1985
#Peppoloni, Lorenzo & Lawrence Characterization of the disruption of neural control strategies for dynamic fingertip forces from attractor reconstruction


1. Check stationarity
2. Estimate the time delay tau via autocorrelation
3. Estimate the embedding dimension via FNN

"""
from nolitsa import data, dimension
from statsmodels.tsa.stattools import kpss, adfuller 
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import numpy as np
from Modules.support_functions import *

"1. Check for stationarity of the time series 
adf_test(shizophrenia_p[1])
kpss_test(shizophrenia_p[1])
"Time Series is stationary

"2. Estimate the time delay with autocorrelation
plot_acf(shizophrenia_p[1], lags= 200)
plt.show()


x = data.henon(length=5000)[:, 0]
dim = np.arange(8, 15 + 2)
f1, f2, f3 = dimension.fnn(np.array(shizophrenia_p[15]), tau=100, dim=dim, window=10, metric='euclidean', maxnum=500)
#f1, f2, f3 = dimension.fnn(x, tau=1, dim=dim, window=10, metric='cityblock', maxnum=None)


plt.title(r'FNN for Henon map')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()

plt.show()


import csv
from itertools import zip_longest


with open("Control_ts.csv","w+") as f:
    writer = csv.writer(f)
    for values in zip_longest(*shizophrenia_c):
        writer.writerow(values)