#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:11:36 2020

@author: matthiasboeker

"""

"Initial Parameters for the Gaussian Mixture HMM"
N = 3
X = np.array(shizophrenia_p[3][:1000])
T = len(X)
X = X.reshape(len(X), 1)

#Initialize the Gaussian Mixture HMM 
model = hmm.GaussianHMM(n_components=N,  covariance_type="spherical", algorithm='viterbi', random_state=0,n_iter=100)
model._init(X)
# Estimate the b 
b = model._compute_log_likelihood(X)

#Estimate the alphas and betas from forward and backward algo 
alpha_sum , log_alphas = model._do_forward_pass(b)
log_betas = model._do_backward_pass(b)



'Test version'
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

        for i in range(n_components):
            for j in range(n_components):
                work_buffer[i, j] = (fwdlattice[t, i]
                                            #make transition matrix dependent on time 
                                         + log_transmat[i, j]
                                         + framelogprob[t + 1, j]
                                         + bwdlattice[t + 1, j]
                                         - logprob[t] - div[t])
        log_xi[:,:,t] = work_buffer[:, :]
    return(log_xi)

w_res = _calc_xi(T,N,log_alphas, log_mask_zero(model.transmat_),log_betas, b)

'Working version'
def _compute_log_xi(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob, log_xi):
    work_buffer = np.full((n_components, n_components), -np.inf, dtype= np.float64)
    #logprob = np.zeros(n_samples-1) 
    logprob = np.zeros(T-1)
    #log_xi_sum = np.zeros((n_components, n_components, n_samples))
    log_xi_sum = np.full((n_components, n_components, n_samples), -np.inf, dtype= np.float64)
    for t in range(n_samples - 1):
        logprob[t] = special.logsumexp(fwdlattice[t,:])
        #logprob[t] = special.logsumexp(fwdlattice[t,:])
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[i, j] = (fwdlattice[t, i]
                                         + log_transmat[i, j]
                                         + framelogprob[t + 1, j]
                                         + bwdlattice[t + 1, j]
                                         - logprob[t])

        for i in range(n_components):
            for j in range(n_components):
                log_xi[i, j] = np.logaddexp(log_xi[i, j],work_buffer[i, j])
        log_xi_sum[:,:,t] = log_xi[:, :]
    return(log_xi_sum)

w_res = _compute_log_xi(T,N,log_alphas, log_mask_zero(model.transmat_),log_betas, b,log_xi_sum)


                                             
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



log_xi = np.full((N, N), -np.inf)
w_res = _compute_log_xi(T,N,log_alphas, log_mask_zero(model.transmat_), b, log_betas, log_xi)
