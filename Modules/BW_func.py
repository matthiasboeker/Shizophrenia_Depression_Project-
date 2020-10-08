#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:46:03 2020

@author: matthiasboeker
Baum-Welch Functions
"""
import matplotlib.pyplot as plt
from scipy import special 
from scipy import optimize
import numpy as np
from Modules.help_functions import *



def _viterbi(n_samples, n_components, log_startprob, log_transmat, framelogprob):


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






#Object funtion for the maximization of the transition ML 
def object_fun(x,T,Z,Xi,N,ind):
    #print('optimization in process')

    x = x.reshape((N,N,ind+1))
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
    logprob = np.zeros(n_samples-1)
    div = np.zeros(n_samples-1)
    #log_xi_sum = np.zeros((n_components, n_components, n_samples))
    log_xi = np.full((n_components, n_components, n_samples), -np.inf, dtype= np.float64)
    
    for t in range(n_samples - 1):
        logprob[t] = special.logsumexp(fwdlattice[t,:])
        div[t] = special.logsumexp(special.logsumexp(log_transmat[:,:,t],axis=0)+framelogprob[t + 1, :]+bwdlattice[t + 1, :])
        work_buffer = _calc_xi_step(n_samples,n_components,fwdlattice[t,:], log_transmat[:,:,t],bwdlattice[t,:], framelogprob[t,:])
        
        
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
    log_prob = special.logsumexp(buffer[-1])
    return(log_prob, buffer)


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
 
def convergence(hist, breake, tol):
    latest = len(hist)
    if latest > 2: 
        delta = hist[latest-1] -  hist[latest-2]
        if delta < tol :
            breake = True 
  


    
def EA_func(model, N, X, Z, tol, cycles, bnds, ind):
    #1.4 Initialize the model 
    T = len(X)
    model.transmat_ = np.repeat(model.transmat_ [:,:,np.newaxis], T, axis = 2)
    
    #Set up logprob history for convergence 
    hist = []
    breake = False
    
    for cyc in cycles:
        print('Running in iteration: ', cyc)
        #2.0 Expectation Step
        #2.1 First time calculating the b, forward algo alphas, backward algo betas
        #Remember for the log_likelihood update model.means_ and model._covars
        b = model._compute_log_likelihood(X)
        
        #Estimate the alphas and betas from forward and backward algo 
        log_prob, log_alphas = time_forward( T,N, log_mask_zero(model.startprob_), log_mask_zero(model.transmat_), b)
        log_betas = time_backward(T, N, log_mask_zero(model.startprob_), log_mask_zero(model.transmat_), b)
        
        hist.append(log_prob)
        convergence(hist, breake, tol)
        if breake == True:
            print('Logprob not increasing')
            print('iteration: ', cyc)
            break
        
        
        #Calculate the Xis 
        Xi = _calc_xi(T,N,log_alphas, log_mask_zero(model.transmat_),log_betas, b)
        
        gamma = _calc_gamma(log_alphas, log_betas)
        
        #2.2 Optimization step
        #Initialize for the optimization 
        
        #Initialize first solution guess
        x0 = np.zeros([N,N,ind+1])
        x0[:,:,:ind+1] = np.random.uniform(0,1,[N,N,ind+1])  
        
        
        #Minimize object function to get the optimized covariate coefficients 
        param = (T, Z, Xi,N, ind)
        options = {'maxiter':1000}
        print('Start of internal optimization')
        res = optimize.minimize(object_fun,x0 = x0, bounds=bnds,options=options,args=param,method='SLSQP')
        res_x = res.x.reshape((N,N,ind+1))
        print('End of internal optimization')
        #Update the intital guess 
        #x0 = res_x
        
        #Calculate the time dependent Transmittion probability 
        trans_ = calc_trans(T,N, Xi, res_x, Z)
        
        #3 Do M step 
        #3.1 Update 
        model.startprob_, model.transmat_ = m_step(trans_, gamma)
        
        
        #Calculate the new covariance and means 
        model.means_means, model.covars_ = _calc_mean_cov(gamma, X, model.means_prior, model.means_weight, model.covars_prior, model.covars_weight)


       