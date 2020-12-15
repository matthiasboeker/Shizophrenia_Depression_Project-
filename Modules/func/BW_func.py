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
from Modules.func.help_functions import *
from hmmlearn import stats
import matplotlib.pyplot as plt

def load_boundaries(N,ind):
        #Introduce boundaries for the optimization, set diag zero
    lb = -np.inf*np.ones([N,N,ind+1])
    [np.fill_diagonal(lb[:,:,l],0) for l in range(0,lb.shape[2])]
    ub = np.inf*np.ones([N,N,ind+1])
    [np.fill_diagonal(ub[:,:,l],0) for l in range(0,ub.shape[2])]
    bnds = optimize.Bounds(lb.flatten(),ub.flatten())
    return bnds         


#Object funtion for the maximization of the transition ML 
def object_fun(x,T,Z,Xi,N,ind):
    #print('call object func')

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




#Calculate the overall time dependent log Xi   
def _calc_xi(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob, log_gamma):
    work_buffer = np.full((n_components, n_components, n_samples-1), -np.inf,dtype= np.float64)    
    for t in range(0,n_samples - 1):
        for i in range(0,n_components):
            for j in range(0,n_components):
                work_buffer[i, j, t] = log_gamma[t,i] + log_transmat[i, j, t]+ framelogprob[t+1, j] + bwdlattice[t+1, j] - bwdlattice[t, i]
    return(work_buffer)







#"Running versions"
#def _calc_xi_step(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob):
#    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
#    logprob = special.logsumexp(fwdlattice)
#    div = special.logsumexp(special.logsumexp(log_transmat,axis=0)+framelogprob+bwdlattice)
#    for i in range(n_components):
#        for j in range(n_components):
#            work_buffer[i, j] = (fwdlattice[i]
#                                         + log_transmat[i, j]
#                                         + framelogprob[j]
#                                         + bwdlattice[j]
#                                         - logprob - div)
#    return(work_buffer)

#Calculate the overall time dependent log Xi   
#def _calc_xi(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob):
#    
#    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
#    #logprob = np.zeros(n_samples-1) 
#    logprob = np.zeros(n_samples-   1)
#    div = np.zeros(n_samples-1)
#    #log_xi_sum = np.zeros((n_components, n_components, n_samples))
#    log_xi = np.full((n_components, n_components, n_samples), -np.inf, dtype= np.float64)
#    
#    for t in range(n_samples - 1):
#        logprob[t] = special.logsumexp(fwdlattice[t,:])
#        #div[t] = special.logsumexp((special.logsumexp(log_transmat[:,:,t], axis=0),framelogprob[t + 1, :],bwdlattice[t + 1, :]))
#        work_buffer = _calc_xi_step(n_samples,n_components,fwdlattice[t,:], log_transmat[:,:,t],bwdlattice[t+1 ,:], framelogprob[t+1,:])
#        
#        
#        log_xi[:,:,t] = work_buffer[:, :]
#    return(log_xi)




"Time dependent Transition prob Foward Algorithm"

def time_forward( n_samples,n_components, log_startprob, log_transmat, framelogprob):

    fwdlattice = np.zeros((n_samples, n_components), dtype= np.float64 )
    work_buffer = np.full(n_components, -np.inf,dtype= np.float64)
    for i in range(n_components):
        fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]
    for t in range(1, n_samples):
        for j in range(n_components):
            for i in range(n_components):

                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j, t-1]
            fwdlattice[t, j] = special.logsumexp(work_buffer) + framelogprob[t, j]
                        
    log_prob = special.logsumexp(fwdlattice[-1])
    return(log_prob, fwdlattice)




def time_backward(n_samples, n_components, log_startprob, log_transmat, framelogprob):
    work_buffer = np.full(n_components,-np.inf, dtype= np.float64 )
    bwdlattice = np.full((n_samples, n_components), -np.inf, dtype= np.float64 )
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
                

def _calc_gamma(n_samples, n_components, log_alpha, log_beta):
    log_gamma = log_alpha + log_beta
    log_normalize(log_gamma, axis = 1)
    return(log_gamma)
    

def _calc_mean_cov(posteriors, obs, means_prior, means_weight, covars_prior, covars_weight):
    
    #Extract mean and covariance
    posteriors = np.exp(posteriors)
    post = posteriors.sum(axis=0)
    obs_m = np.dot(posteriors.T, obs)
    obs_c = np.dot(posteriors.T, obs **2 )
    
    denom = post[:, None]
    
    means = ((means_weight*means_prior + obs_m))/(means_weight + denom)
    
    
    mean_diff = means -  means_prior
    
    cv_num = (means_weight * mean_diff**2 + obs_c - 2 * means * obs_m + means**2 * denom)

    cv_den = max(covars_weight - 1, 0) + denom
    covars = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    
    return means, covars
 

# ERROR IN CACULATION HERE PRODUCES NANs
def calc_trans_w(n_samples,n_components ,log_gamma , Xi):
    trans_ = np.full((n_components,n_components,n_samples-1), -np.inf, dtype= np.float64 )
    for t in range(0,1440):
        for i in range(0,n_components):
            for j in range(0,n_components):
                trans_[i,j,t] = Xi[i,j,t]- log_gamma[t,i] 

    return trans_



def calc_trans(n_samples,n_components ,  Xi, res_x, Z, gamma ):
    trans_ = np.zeros((n_components,n_components,n_samples-1))
    for t in range(0,n_samples-1):
        for i in range(0,n_components):
            for j in range(0,n_components):
                #trans_[i,j,t] = np.exp(Xi[i,j,t])*((res_x[i,j,0]+np.dot(res_x[i,j,1],Z[t]))-special.logsumexp(res_x[i,:,0]+np.dot(res_x[i,:,1],Z[t])))/gamma[t,i]
                trans_[i,j,t] = (res_x[i,j,0]+np.dot(res_x[i,j,1],Z[t]))-special.logsumexp(res_x[i,:,0]+np.dot(res_x[i,:,1],Z[t]))
    
    return trans_


def m_step(trans_, posteriors):
    
    startprob_ = posteriors[0]
    startprob_ = np.where(startprob_ == 0, 1e-10, startprob_)
    normalize(startprob_)
    transmat_ = np.where(trans_ == 0, 1e-10, trans_)
    normalize(transmat_, axis=1)
    return startprob_, transmat_
 
def convergence(hist, breake, tol):
    latest = len(hist)
    if latest > 2: 
        delta = hist[latest-1] -  hist[latest-2]
        if delta < tol :
            breake = True 
  
    
    
def EA_func(model, N, X, Z, cycles, bnds, ind,  tol=1e-5):
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

        log_prob, log_alphas = time_forward( T,N, log_mask_zero(model.startprob_), log_mask_zero(model.transmat_), b)
        log_betas = time_backward(T, N, log_mask_zero(model.startprob_), log_mask_zero(model.transmat_), b)
        if np.isinf(log_alphas).any():
            print('alphas inf')
            print(np.argwhere(np.isinf(log_alphas)))


        
        hist.append(log_prob)
        convergence(hist, breake, tol)
        if breake == True:
            print('Logprob not increasing')
            print('iteration: ', cyc)
            break
        
        gamma = _calc_gamma(T, N, log_alphas, log_betas)

        #Calculate the Xis 
        Xi = _calc_xi(T,N,log_alphas, log_mask_zero(model.transmat_),log_betas, b, gamma)

        #2.2 Optimization step
        #Initialize for the optimization 
        
        #Initialize first solution guess
        x0 = np.zeros([N,N,ind+1])
        x0[:,:,:ind+1] = np.random.uniform(0,1,[N,N,ind+1])  
        
        
        #Minimize object function to get the optimized covariate coefficients 
        param = (T, Z, Xi,N, ind)
        options = {'maxiter':50}
        #print('Start of internal optimization')
        #res = optimize.minimize(object_fun,x0 = x0, bounds=bnds,options=options,args=param,method='SLSQP')
        #res_x = res.x.reshape((N,N,ind+1))
        #print('End of internal optimization')
        #Update the intital guess 
        #print('Link Coefficients',res_x )
        #x0 = res_x
        
        #Calculate the time dependent Transmittion probability with or without Covarate
        #trans_ = calc_trans(T,N, Xi, x0, Z, gamma) #with Covariate
        
        trans_ = calc_trans_w(T,N ,gamma , Xi)
        


        #3 Do M step 
        #3.1 Update 
        model.link_coef = x0 
        model.startprob_, model.transmat_ = m_step(np.exp(Xi), np.exp(gamma))
        
        
        #print(np.array([[np.mean(model.transmat_[0,0,:]),np.mean(model.transmat_[0,1,:])],[np.mean(model.transmat_[1,0,:]),np.mean(model.transmat_[1,1,:])]]))
        #Calculate the new covariance and means 
        model.means_, model.covars_ = _calc_mean_cov(gamma, X, model.means_prior, model.means_weight, model.covars_prior, model.covars_weight)
        print(np.mean(model.transmat_,axis=2))
    model.log_prob = hist