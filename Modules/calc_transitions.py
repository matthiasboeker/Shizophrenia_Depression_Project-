#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:17:24 2020

@author: matthiasboeker
"""
from scipy import optimize
from scipy import special
import numpy as np

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


#Calculate the time-varying transition matrix
#Initial Covariates Xt
days = (len(X))/60/24
z = np.arange(0,T)
#one covariate
Z = np.array(np.cos(days*2*np.pi*z/T)+1)




"initial guess"
"ind plus one for the constant c0"
"Make it state dependent

x0 = np.zeros([N,N,ind+1])
x0[:,:,:ind+1] = np.random.uniform(0,1,[N,N,ind+1])  
lb = -np.inf*np.ones([N,N,ind+1])
[np.fill_diagonal(lb[:,:,l],0) for l in range(0,lb.shape[2])]
ub = np.inf*np.ones([N,N,ind+1])
[np.fill_diagonal(ub[:,:,l],0) for l in range(0,ub.shape[2])]
bnds = optimize.Bounds(lb.flatten(),ub.flatten())

"Calculate the Xis 
Xi = _calc_xi(T,N,log_alphas, log_mask_zero(model.transmat_),log_betas, b)


"Optimization of Transition Matrix"
#minimize object function 
param = (T,Z, Xi,N)
res = optimize.minimize(object_fun,x0 = x0, bounds=bnds,maxiter=1000,args=param,method='SLSQP')
res_x = res.x.reshape((3,3,2))

    
"Calculate the Transition coefficients"
trans_ = np.zeros((N,N,T-1))
"Function in work"
for t in range(0,T-1):
    for i in range(0,N):
        for j in range(0,N):
            trans_[i,j,t] = np.exp(Xi[i,j,t])*((res_x[i,j,0]+np.dot(res_x[i,j,1],Z[t]))-special.logsumexp(res_x[i,:,0]+np.dot(res_x[i,:,1],Z[t])))
                
"Get Transition matrix dependent on time: Normalize the Transition coefficients"
transmat_ = np.where(trans_ == 0, 0, trans_)
normalize(transmat_, axis=1)

"Get the covariance and mean"
model = hmm.GaussianHMM(n_components=3,  covariance_type="full", algorithm='viterbi', params='s',random_state=0,n_iter=100)
model.trans
model.fit(X)
model.cov