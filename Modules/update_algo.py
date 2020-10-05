#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:04:07 2020

@author: matthiasboeker
"""

from hmmlearn import hmm
from hmmlearn import _hmmc
from scipy import special
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt

def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)

def normalize(a, axis=None):
    """
    Normalizes the input array so that it sums to 1.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum


def log_normalize(a, axis=None):
    """
    Normalizes the input array so that ``sum(exp(a)) == 1``.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = logsumexp(a, axis, keepdims=True)
        a -= a_lse


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

"Initial Parameters for the Gaussian Mixture HMM"
N = 3
X = np.array(shizophrenia_p[3][:1000])
T = len(X)
X = X.reshape(len(X), 1)

#Initialize the Gaussian Mixture HMM 
model = hmm.GaussianHMM(n_components=N,  covariance_type="full", algorithm='viterbi', random_state=0,n_iter=100)
model._init(X)
# Estimate the b 
b = model._compute_log_likelihood(X)

#Estimate the alphas and betas from forward and backward algo 
alpha_sum , log_alphas = model._do_forward_pass(b)
log_betas = model._do_backward_pass(b)

log_xi_sum = np.full((N, N), -np.inf)
trans = np.zeros((N, N))


trans = np.zeros((3,3))

trans += np.exp(w_res[:,:,T-2])

transmat_ = np.maximum(model.transmat_prior - 1 + trans, 0)
transmat_ = np.where(transmat_ == 0, 0, transmat_)
normalize(transmat_, axis=1)