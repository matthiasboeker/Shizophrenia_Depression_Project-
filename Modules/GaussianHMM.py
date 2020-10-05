#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:51:04 2020

@author: matthiasboeker
Forward-Backward Algorithm 
"""

from hmmlearn import hmm
import numpy as np 
from scipy import optimize
from scipy import special
import math
from scipy.optimize import Bounds


def divide(X, n):
    sort = np.argsort(X)
    idx = np.array_split(sort, n)
    return(idx)
    
def object_fun(x,T,Z,Xi,N):
    print('function call process')

    x = x.reshape((3,3,2))
    print(x[:,:,1])
    temp = np.zeros((T-1)*N*N)
    c=0
    for t in range(0,T-1):
        for i in range(0,N):
            for j in range(0,N):
                temp[c] = Xi[i,j,t]*((x[i,j,0]+np.dot(x[i,j,1],Z[t]))-special.logsumexp(x[i,:,0]+np.dot(x[i,:,1],Z[t])))
                c=c+1
                
    f=10000/np.sum(temp)
    return f

#Find the initial means and covariance matrices for each of the states
# Split the observations into evenly size states from smallest to largest

"Initial Parameters for the Gaussian Mixture HMM"
N = 3
X = np.array(shizophrenia_p[3][:1000])
T = len(X)
X = X.reshape(len(X), 1)
idx = divide(X,N)

mu = np.zeros(N)
cov = np.zeros(N)
for i in range(0,N):
    mu[i] = np.mean(X[idx[i]])
    cov[i] = np.var(X[idx[i]])

#Initial random Priors 
pi = np.random.uniform(0,1,N)

#Initialize weights 
weights = np.random.uniform(0,1,N)

#Initial Transition matrix 
A = np.random.rand(N,N)



#Initialize the Gaussian Mixture HMM 
model = hmm.GaussianHMM(n_components=N,  covariance_type="full", algorithm='viterbi', random_state=0,n_iter=100)
model._init(X)
# Estimate the b 
b = model._compute_log_likelihood(X)

#Estimate the alphas and betas from forward and backward algo 
alpha_sum , log_alphas = model._do_forward_pass(b)
log_betas = model._do_backward_pass(b)

#Transition matrix
A = [model.transmat_]*T
log_A = np.log(model.transmat_)
log_A = [log_A]*T
alphas = np.exp(log_alphas)
betas = np.exp(log_betas)
exp_b = np.exp(b)


# E The expectation step estimate the state occupation probabilities 
Xi = np.zeros((N, N, T - 1))
for t in range(0,T-1):
    for i in range(0,N):
        for j in range(0,N):
            Xi[i,j,t] = log_alphas[t,i]+log_A[t][i,j]+b[t+1,j]+log_betas[t+1,j]
    d = Xi[:,:,t]
    Xi[:,:,t] = d - special.logsumexp(Xi[:,:,t])



# E The expectation step estimate the state occupation probabilities 
Xi = np.zeros((N, N, T - 1))
for t in range(0,T-1):
    for i in range(0,N):
        for j in range(0,N):
            Xi[i,j,t] = (alphas[t,i]*A[t][i,j]*exp_b[t+1,j]*betas[t+1,j])/special.logsumexp(Xi[:,:,t])



Gamma = np.sum(Xi,axis=1)

# M The maximization step: Re-estimate the HMM parameters
Pi = Gamma[:,1]

#Calculate the time-varying transition matrix
#Initial Covariates Xt
days = (len(X))/60/24
z = np.arange(0,T)
#one covariate
ind = 1
Z = np.array(np.cos(days*2*np.pi*z/T)+1)



#initial guess
#ind plus one for the constant c0
x0 = np.zeros([N,N,ind+1])
x0[:,:,:ind+1] = np.random.uniform(0,1,[N,N,ind+1])  
lb = -math.inf*np.ones([N,N,ind+1])
[np.fill_diagonal(lb[:,:,l],0) for l in range(0,lb.shape[2])]
ub = math.inf*np.ones([N,N,ind+1])
[np.fill_diagonal(ub[:,:,l],0) for l in range(0,ub.shape[2])]
bnds = Bounds(lb.flatten(),ub.flatten())


"Optimization of Transition Matrix"
#minimize object function 
param = (T,Z, Xi,N)
res = optimize.minimize(object_fun,x0 = x0, bounds=bnds,args=param,method='SLSQP')
res_x = res.x.reshape((3,3,2))

object_fun(x0,T,Z,Xi,N)

'Calculate the transition matrix 
A = np.zeros((N,N,T))
for t in range(0,T-1):
    for i in range(0,N):
        for j in range(0,N):
            A[i,j,t] = Xi[i,j,t]*((res_x[i,j,0]+np.dot(res_x[i,j,1],Z[t]))-special.logsumexp(res_x[i,:,0]+np.dot(res_x[i,:,1],Z[t])))
 
plt.scatter(range(0,1000),X, marker='.')
plt.plot(range(0,1000),Z*1000)