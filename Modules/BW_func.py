#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:46:03 2020

@author: matthiasboeker
Baum-Welch Functions
"""

def object_fun(x,T,Z,Xi,N):
    print('function call process')

    x = x.reshape((3,3,2))
    print(x[:,:,1])
    temp = np.zeros((T-1)*N*N)
    c=0
    for t in range(0,T-1):
        for i in range(0,N):
            for j in range(0,N):
                temp[c] = np.exp(Xi[i,j,t])*((x[i,j,0]+np.dot(x[i,j,1],Z[t]))-special.logsumexp(x[i,:,0]+np.dot(x[i,:,1],Z[t])))
                c=c+1
                
    f=1000/np.sum(temp)
    return f

"Testable functions"
def _calc_xi_step(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob):
    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
    logprob = special.logsumexp(fwdlattice)
    div = special.logsumexp(special.logsumexp(log_transmat,axis=0)+framelogprob+bwdlattice)
    for i in range(n_components):
        for j in range(n_components):
            work_buffer[i, j] = (fwdlattice[i]
                                         + log_transmat[i, j]
                                         + framelogprob[ j]
                                         + bwdlattice[j]
                                         - logprob - div)
    return(work_buffer)
    
def _calc_xi(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob):
    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
    #logprob = np.zeros(n_samples-1) 
    logprob = np.zeros(T-1)
    div = np.zeros(T-1)
    #log_xi_sum = np.zeros((n_components, n_components, n_samples))
    log_xi = np.full((n_components, n_components, n_samples), -np.inf, dtype= np.float64)
    
    for t in range(n_samples - 1):
        logprob[t] = special.logsumexp(fwdlattice[t,:])
        div[t] = special.logsumexp(special.logsumexp(log_transmat,axis=0)+framelogprob[t + 1, :]+bwdlattice[t + 1, :])
        work_buffer = _calc_xi_step(T,N,log_alphas[t,:], log_mask_zero(model.transmat_),log_betas[t,:], b[t,:])
        
        
        log_xi[:,:,t] = work_buffer[:, :]
    return(log_xi)

