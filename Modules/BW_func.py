#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:46:03 2020

@author: matthiasboeker
Baum-Welch Functions
"""
from scipy import special 
import numpy as np

#Object funtion for the maximization of the transition ML 
def object_fun(x,T,Z,Xi,N,ind):
    print('function call process')

    x = x.reshape((N,N,ind+1))
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

#Calculate the log Xi per time step 
"Calculate Xi"
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

#Calculate the overall time dependent log Xi     
def _calc_xi(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob):
    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
    
    #logprob = np.zeros(n_samples-1) 
    logprob = np.zeros(T-1)
    div = np.zeros(T-1)
    #log_xi_sum = np.zeros((n_components, n_components, n_samples))
    log_xi = np.full((n_components, n_components, n_samples), -np.inf, dtype= np.float64)
    
    for t in range(n_samples - 1):
        logprob[t] = special.logsumexp(fwdlattice[t,:])
        div[t] = special.logsumexp(special.logsumexp(log_transmat[:,:,t],axis=0)+framelogprob[t + 1, :]+bwdlattice[t + 1, :])
        work_buffer = _calc_xi_step(T,N,log_alphas[t,:], log_mask_zero(model.transmat_[:,:,t]),log_betas[t,:], b[t,:])
        
        
        log_xi[:,:,t] = work_buffer[:, :]
    return(log_xi)

"Version in work!!!"  
def _calc_xi_step_(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob):
    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
    logprob = special.logsumexp(fwdlattice)
    div = special.logsumexp(special.logsumexp(log_transmat[:,:,t],axis=0)+framelogprob+bwdlattice)
    for i in range(n_components):
        for j in range(n_components):
            work_buffer[i, j] = (fwdlattice[i]
                                         + log_transmat[i, j, t]
                                         + framelogprob[ j]
                                         + bwdlattice[j]
                                         - logprob - div)
    return(work_buffer)  
def _calc_xi_(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob):
    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
    #logprob = np.zeros(n_samples-1) 
    logprob = np.zeros(T-1)
    div = np.zeros(T-1)
    #log_xi_sum = np.zeros((n_components, n_components, n_samples))
    log_xi = np.full((n_components, n_components, n_samples), -np.inf, dtype= np.float64)
    
    for t in range(n_samples - 1):
        logprob[t] = special.logsumexp(fwdlattice[t,:])
        div[t] = special.logsumexp(special.logsumexp(log_transmat[:,:,t],axis=0)+framelogprob[t + 1, :]+bwdlattice[t + 1, :])
        work_buffer = _calc_xi_step(T,N,log_alphas[t,:], log_mask_zero(model.transmat_[:,:,t]),log_betas[t,:], b[t,:])
        
        
        log_xi[:,:,t] = work_buffer[:, :]
    return(log_xi)


"Time dependent Transition prob Foward Algorithm"

def time_forward( n_samples,n_components, log_startprob, log_transmat, framelogprob):
    
    fwdlattice = np.zeros((n_samples, n_components))
    work_buffer = np.zeros(n_components, dtype= np.float64)
    buffer = np.zeros((n_samples, n_components))
    for i in range(n_components):
        fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]
        for t in range(1, n_samples-1):
            for j in range(n_components):
                for i in range(n_components):
                    work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j, t]

                buffer[t, j] = special.logsumexp(work_buffer) + framelogprob[t, j]
    return(buffer)


def time_backward(n_samples, n_components, log_startprob, log_transmat, framelogprob):
    work_buffer = np.zeros(n_components, dtype= np.float64)
    bwdlattice = np.zeros((n_samples, n_components))
    for i in range(n_components):
        bwdlattice[n_samples - 1, i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[i, j, t]
                                      + framelogprob[t + 1, j]
                                      + bwdlattice[t + 1, j])
                bwdlattice[t, i] = special.logsumexp(work_buffer)
    return(bwdlattice)
                

def _calc_gamma(log_alpha, log_beta):
    log_gamma = log_alpha + log_beta
    log_normalize(log_gamma, axis=1)
    return(np.exp(log_gamma))
    

def _calc_mean_cov(posteriors, obs, means_prior, means_weight, covars_prior, covars_weight):
    #Extract mean and covariance
    post = posteriors.sum(axis=0)
    obs = np.dot(posteriors.T, obs)
    
    denom = post[:, None]
    
    means = ((means_weight*means_prior + obs))/(means_weight + denom)
    mean_diff = means -  means_prior
    
    cv_num = (means_weight * mean_diff**2 + obs**2 - 2 * means * obs + means**2 * denom)

    cv_den = max(covars_weight - 1, 0) + denom
    covars = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    
    return means, covars
    
def calc_trans(n_samples,n_components ,  Xi, res_x, Z ):
    trans_ = np.zeros((n_components,n_components,n_samples-1))
    for t in range(0,n_samples-1):
        for i in range(0,n_components):
            for j in range(0,n_components):
                trans_[i,j,t] = np.exp(Xi[i,j,t])*((res_x[i,j,0]+np.dot(res_x[i,j,1],Z[t]))-special.logsumexp(res_x[i,:,0]+np.dot(res_x[i,:,1],Z[t])))
    return trans_

def m_step(trans_, posteriors):
    startprob_ = posteriors[0]
    startprob_ = np.where(startprob_ == 0, 0, startprob_)
    normalize(startprob_)
    transmat_ = np.where(trans_ == 0, 0, trans_)
    normalize(transmat_, axis=1)
    return startprob_, transmat_
        