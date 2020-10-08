#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:46:20 2020

@author: matthiasboeker
main Baum Welch Algorithm process
"""
import os
os.chdir('/Users/matthiasboeker/Desktop/Master_Thesis/Schizophrenia_Depression_Project/')
import numpy as np 
from scipy import special 
from scipy import optimize 
from Modules.support_functions import *
from Modules.help_functions import *
from Modules.BW_func import *
from hmmlearn import hmm
from hmmlearn import _hmmc

#1. The initialization
#1.1 load the data
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
#Import days 
days = pd.read_csv('days.csv')
shizophrenia_p, shizophrenia_c = preprocess(days,shizophrenia_p, shizophrenia_c)


#1.2 set external parameters 
N = 2
X = np.array(shizophrenia_p[3][:2500])
T = len(X)
X = X.reshape(T, 1)
algo = 'viterbi'
cycles = range(0,1) 
tol = 1e-5

#1.3 Initial Covariates Xt
days = (len(X))/60/24
z = np.arange(0,T)
#One covariate 
ind = 1
#The modelled Covariate 
Z = np.array(np.cos(days*2*np.pi*z/T)+1)



#Introduce boundaries for the optimization, set diag zero
lb = -np.inf*np.ones([N,N,ind+1])
[np.fill_diagonal(lb[:,:,l],0) for l in range(0,lb.shape[2])]
ub = np.inf*np.ones([N,N,ind+1])
[np.fill_diagonal(ub[:,:,l],0) for l in range(0,ub.shape[2])]
bnds = optimize.Bounds(lb.flatten(),ub.flatten())

#1.4 Initialize the model 
model = hmm.GaussianHMM(n_components=N,  covariance_type="diag", algorithm=algo, random_state=0,n_iter=100)
#Initialize weights and priors for mean, cov, start_probs, transmat
model._init(X)
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
    res = optimize.minimize(object_fun,x0 = x0, bounds=bnds,options=options,args=param,method='SLSQP')
    res_x = res.x.reshape((N,N,ind+1))
    #Update the intital guess 
    #x0 = res_x
    
    #Calculate the time dependent Transmittion probability 
    trans_ = calc_trans(T,N, Xi, res_x, Z)
    
    #3 Do M step 
    #3.1 Update 
    model.startprob_, model.transmat_ = m_step(trans_, gamma)
    
    #Calculate the new covariance and means 
    model.means_means, model.covars_ = _calc_mean_cov(gamma, X, model.means_prior, model.means_weight, model.covars_prior, model.covars_weight)
