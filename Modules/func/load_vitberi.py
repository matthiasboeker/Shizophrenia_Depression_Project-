#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:29:52 2020

@author: matthiasboeker
1. Function for the viterbi algorithm
2. Function to apply the viterbi 

"""
import numpy as np
import os
from hmmlearn import stats

os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project')
from Modules.func.BW_func import *
from Modules.func.support_functions import *

def viterbi(n_samples, n_components,X ,log_startprob, log_transmat, means, covars):


    covars = covars.reshape(n_components,1)
    means = means.reshape(n_components,1)
    #From hmmlearn import stats: to get a multivariaite density for state probabilities
    #Here only the diagonals of the covariance is used! You might want to use other forms
    framelogprob = stats._log_multivariate_normal_density_diag(X, means, covars)
    state_sequence = np.empty(n_samples, dtype=np.int32)
    viterbi_lattice = np.zeros((n_samples, n_components))

    work_buffer = np.empty(n_components)
    for i in range(n_components):
        viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]

        # Induction
    for t in range(1, n_samples-1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[j] = (log_transmat[j, i, t] + viterbi_lattice[t - 1, j])

            viterbi_lattice[t, i] = np.max(work_buffer) + framelogprob[t, i]

        # Observation traceback
    state_sequence[n_samples - 1] = where_from = np.argmax(viterbi_lattice[n_samples - 1])
    logprob = viterbi_lattice[n_samples - 1, where_from]

    for t in range(n_samples - 2, -1, -1):
        for i in range(n_components):
            work_buffer[i] = (viterbi_lattice[t, i] + log_transmat[i, where_from, t])

        state_sequence[t] = where_from = np.argmax(work_buffer)

    return np.asarray(state_sequence), logprob



def load_state_sequence(X, entity):

    entity.state_seq,_ = viterbi(len(X), entity.components ,X ,np.log(entity.start_prob),
             np.log(entity.trans_mat), entity.means,  entity.cov)
