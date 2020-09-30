#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:32:56 2020

@author: matthiasboeker
"""

from hmmlearn import hmm
import numpy as np 

    
#Find the initial means and covariance matrices for each of the states
# Split the observations into evenly size states from smallest to largest

"Initial Parameters for the Gaussian Mixture HMM"
N = 2
X = np.array(shizophrenia_p[3][:1000])
T = len(X)
X = X.reshape(len(X), 1)


#Initialize the Gaussian Mixture HMM 
model = hmm.GaussianHMM(n_components=N,  covariance_type="full", algorithm='viterbi', random_state=0,n_iter=100)
hmm1 = model.fit(X)
probs = model.predict_proba(X)

plt.scatter(range(0,1000),X, marker='.')
plt.plot(range(0,1000),Z*1000)