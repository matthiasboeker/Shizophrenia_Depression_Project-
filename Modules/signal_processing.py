#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:16:43 2020

@author: matthiasboeker
"""
import sklearn.mixture as mix
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from matplotlib import cm
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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



# Calculate the AICS and BICS for different d parameteters for different individuals, time delay embedding Conditioned 
series = range(0,len(shizophrenia_c))
frames = [gaussian_filter(shizophrenia_c[n], sigma=50) for n in series]
params = range(1,10)

T = [[np.array(frame)[np.arange(w)+ np.arange(np.max(frame.shape[0] - (w-1), 0)).reshape(-1,1)] for w in params ] for frame in frames ]

model =  mix.GaussianMixture(4, covariance_type='full', random_state=0)
aics = [[model.fit(matrix).aic(matrix) for matrix in matrices[k]] for k in range(0,32)]
bics = [[model.fit(matrix).bic(matrix) for matrix in matrices[k]] for k in range(0,32)]
#Plot the AIC
for i in range(0,len(aics)):
    plt.plot(range(0,9), aics[i])
    plt.ylabel('AIC')
    plt.xlabel('d parameter')
    
#Plot the AIC
for i in range(0,len(aics)):
    plt.plot(range(0,9), aics[i])
    plt.ylabel('BIC')
    plt.xlabel('d parameter')
    
    
    

# Calculate different sigmas for the Gaussian filter to find empirical optimum 
# Dependency on the choice of states? When states robust maybe sigmas also but sigma must be more than 10? 
#Sparsity/ zeros leads to a vertical clustering!!!
    #Explanation? change in the gaussian mixture to model mainly the mean, not the variance??
    #BIC does not cover that in the total number but maybe in the edge at 10

#Patients
series = range(0,len(shizophrenia_p))
sigmas = range(1,11)
arrays = [[gaussian_filter(shizophrenia_p[n], sigma=sigma) for sigma in sigmas]for n in series] 

model =  mix.GaussianMixture(4, covariance_type='spherical', random_state=1)
bics_p = [[model.fit(array.reshape(-1, 1)).bic(array.reshape(-1, 1)) for array in arrays[k]] for k in range(0,len(shizophrenia_p))]

    
#Plot the BIC for sigma per individual 
for i in range(0,len(bics_p)):
    plt.plot(range(1,11), bics_p[i])
    plt.ylabel('BIC')
    plt.xlabel('Sigma')


#Control 
series = range(0,len(shizophrenia_c))
sigmas = range(1,11)
arrays = [[gaussian_filter(shizophrenia_c[n], sigma=sigma) for sigma in sigmas]for n in series] 

model =  mix.GaussianMixture(4, max_iter=500 ,covariance_type='spherical', random_state=10)
bics_c = [[model.fit(array.reshape(-1, 1)).bic(array.reshape(-1, 1)) for array in arrays[k]] for k in range(0,len(shizophrenia_c))]

#Very important Plot: the control group reaches the sigma edge earlier than the patient group
#For a fit the gaussian filter has to allow more varation in filter to get a horizontal clustering! 
#How can we use it? 
fig, (ax1,ax2) = plt.subplots(1,2,sharex=True, figsize=(15, 7))
for i in range(0,len(bics_p)):
    ax1.plot(range(1,11), bics_p[i])
    ax1.set_ylabel('BIC')
    ax1.set_xlabel('Sigma')
    ax1.set_title('Patients')
    ax2.plot(range(1,11), bics_c[i])
    ax2.set_ylabel('BIC')
    ax2.set_xlabel('Sigma')
    ax2.set_title('Control')




#Patient
patient = np.array(shizophrenia_p[1]).reshape(-1,1)
sigmas = range(1,11)
n_comp = range(1,11)
models = [mix.GaussianMixture(n, covariance_type='full', random_state=0)
          for n in n_comp]
grid_p = np.array([[model.fit(gaussian_filter(patient, 
                                   sigma=sigma)).bic(gaussian_filter(patient, sigma=sigma)) for sigma in sigmas]for model in models]) 
#Surface Plot Patient 
x = np.arange(1, 11, 1)
y = np.arange(1, 11, 1)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.view_init(elev=30., azim=130)
ax.set_ylabel('States')
ax.set_xlabel('Sigma')
ax.set_zlabel('BIC')
surf = ax.plot_surface(X, Y, grid_p, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


#Control
control = np.array(shizophrenia_c[1]).reshape(-1,1)
sigmas = range(1,11)
n_comp = range(1,11)
models = [mix.GaussianMixture(n, covariance_type='full', random_state=0)
          for n in n_comp]
grid_c = np.array([[model.fit(gaussian_filter(control, 
                                   sigma=sigma)).bic(gaussian_filter(control, sigma=sigma)) for sigma in sigmas]for model in models])

#Surface Plot Patient 
x = np.arange(1, 21, 1)
y = np.arange(1, 21, 1)
X, Y = np.meshgrid(x, y)
Z = grid_c

fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.view_init(elev=30., azim=130)
ax.set_ylabel('States')
ax.set_xlabel('Sigma')
ax.set_zlabel('BIC')
surf = ax.plot_surface(X, Y, grid_c, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()




# Analysis of the GGM weights and their sparsity of patients 
series = range(0,len(shizophrenia_p))
gf = [gaussian_filter(shizophrenia_p[n], sigma=10) for n in series]
model =  mix.GaussianMixture(10, covariance_type='full', random_state=0)
#Calculate and sort the fitted GMM coefficient weights 
gmm_weights_p = [np.sort(model.fit(np.array(patient).reshape(-1,1)).weights_)[::-1] for patient in gf ]

#Plot for every patient
for i in range(0,len(gmm_weights_p)):
    plt.plot(range(1,11), gmm_weights_p[i])
    plt.plot(range(1,11), np.repeat(0.1, 10))
    plt.ylabel('Weight')
    plt.xlabel('Number of coefficient')
#Plot of the average weights 
plt.plot(np.mean(gmm_weights_p, axis=0))
plt.plot(range(0,10), np.repeat(0.1, 10))
plt.ylabel('Weight')
plt.xlabel('Number of coefficient')  

# Analysis of the GGM weights and their sparsity of the control group  
series = range(0,len(shizophrenia_c))
gf = [gaussian_filter(shizophrenia_c[n], sigma=10) for n in series]
model =  mix.GaussianMixture(10, covariance_type='full', random_state=0)
#Calculate and sort the fitted GMM coefficient weights 
gmm_weights_c = [np.sort(model.fit(np.array(control).reshape(-1,1)).weights_)[::-1] for control in gf ]


#Plot for every patient
for i in range(0,len(gmm_weights_c)):
    plt.plot(range(1,11), gmm_weights_c[i])
    plt.plot(range(1,11), np.repeat(0.1, 10))
    plt.ylabel('Weight')
    plt.xlabel('Number of coefficient')
    
#Plot average weights 
plt.plot(np.mean(gmm_weights_c, axis=0))
plt.plot(range(0,10), np.repeat(0.1, 10))
plt.ylabel('Weight')
plt.xlabel('Number of coefficient')  





