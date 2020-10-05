#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:55:23 2020

@author: matthiasboeker
Analysis of the time series distribution 
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import stats

#Import the data 
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


"Analyse the distributions and show that differentiated one works maybe better in jupyter notebook!"
d = 1
frame = shizophrenia_p[0]
frame = shizophrenia_p[0][:60*24].diff(d)[d:]
woz = frame[frame!=0]
plt.scatter(frame.index, frame, marker='.')
plt.scatter(frame.index[:24*60],frame[:24*60], marker='.')
np.log(woz).hist(bins=50)





# Calculate BIC for different embedding dimensions 
d=1
series_p = range(0,len(shizophrenia_p))
frames_p = [shizophrenia_p[n].diff(d)[d:] for n in series_p]
series_c = range(0,len(shizophrenia_c))
frames_c = [shizophrenia_c[n].diff(d)[d:] for n in series_c]


"Differ the overlapping intervals 
w = 60 #window size of w minutes
tau = [15, 30, 45, 60, 75, 90] # overlapping time of the intervals of g minutes 
matrices_p = [[np.array(frame)[np.arange(w)+ np.arange(0,np.max(frame.shape[0] - (w-1), 0), t).reshape(-1,1)]  for t in tau]  for frame in frames_p ] 
matrices_c = [[np.array(frame)[np.arange(w)+ np.arange(0,np.max(frame.shape[0] - (w-1), 0), t).reshape(-1,1)]  for t in tau]    for frame in frames_c ]


model =  mix.GaussianMixture(2, covariance_type='full', random_state=0)
bics_p = [[model.fit(matrix).bic(matrix) for matrix in matrices_p[k]] for k in range(0,len(matrices_p))]
bics_c = [[model.fit(matrix).bic(matrix) for matrix in matrices_c[k]] for k in range(0,len(matrices_c))]
#Plot the AIC
for i in range(0,len(bics_p)):
    plt.plot(range(1,7), bics_p[i])
    plt.ylabel('BIC')
    plt.xlabel('tau parameter')
    
#Plot the AIC
for i in range(0,len(bics_c)):
    plt.plot(range(1,7), bics_c[i])
    plt.ylabel('BIC')
    plt.xlabel('tau parameter')
    

"Differentiate the window intervals 
window = [15, 30, 45, 60, 75, 90] #window size of w minutes
tau = 15 # overlapping time of the intervals of g minutes 
matrices_p = [[np.array(frame)[np.arange(w)+ np.arange(0,np.max(frame.shape[0] - (w-1), 0), tau).reshape(-1,1)]  for w in window]  for frame in frames_p ] 
matrices_c = [[np.array(frame)[np.arange(w)+ np.arange(0,np.max(frame.shape[0] - (w-1), 0), tau).reshape(-1,1)]  for w in window]    for frame in frames_c ]


model =  mix.GaussianMixture(2, covariance_type='full', random_state=0)
bics_p = [[model.fit(matrix).bic(matrix) for matrix in matrices_p[k]] for k in range(0,len(matrices_p))]
bics_c = [[model.fit(matrix).bic(matrix) for matrix in matrices_c[k]] for k in range(0,len(matrices_c))]
#Plot the AIC
for i in range(0,len(bics_p)):
    plt.plot(range(1,7), bics_p[i])
    plt.ylabel('AIC')
    plt.xlabel('window parameter')
    
#Plot the AIC
for i in range(0,len(bics_c)):
    plt.plot(range(1,7), bics_c[i])
    plt.ylabel('BIC')
    plt.xlabel('window parameter')





"Evaluation over window and tau grid 
d = 1
frame = shizophrenia_p[0].diff(d)[d:]

"Differentiate the window intervals "
window = [30, 45, 60] #window size of w minutes
tau = [1,15, 30] # overlapping time of the intervals of g minutes 
matrices_grid = [[np.array(frame)[np.arange(w)+ np.arange(0,np.max(frame.shape[0] - (w-1), 0), t).reshape(-1,1)]  for w in window]  for t in tau ] 


model =  mix.GaussianMixture(2, covariance_type='full', random_state=0)
bics_grid = np.array([[model.fit(matrix).bic(matrix) for matrix in matrices_grid[k]] for k in range(0,len(matrices_grid))])

#Surface Plot Patient 
x = np.arange(1, 4, 1)
y = np.arange(1, 4, 1)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12,10))
ax = fig.gca(projection='3d')
ax.view_init(elev=30., azim=130)
ax.set_ylabel('tau')
ax.set_xlabel('window size')
ax.set_zlabel('BIC')
surf = ax.plot_surface( Y,X, bics_grid, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

matrices = [np.array(frame)[np.arange(10)+ np.arange(np.max(frame.shape[0] - (10-1), 0)).reshape(-1,1)] for frame in frames ]
n_comp = range(1,10)
models = [mix.GaussianMixture(n, covariance_type='full', random_state=0)
          for n in n_comp]
grid_p = np.array([[model.fit(matrix).bic(matrix) for matrix in matrices]for model in models]) 



tt = shizophrenia_p[n]
tt_diff = shizophrenia_p[n].diff(d)[d:]
import sklearn.mixture as mix
model = mix.GaussianMixture(n_components=2, covariance_type='full', 
                                     random_state=0).fit(matrix)

print(model.converged_)
hidden_states = model.predict(matrix)
state_prob = model.predict_proba(matrix)

fig = plt.figure(figsize=(8,5))
color = ['darkorange' if x>2 else 'green' if x==2 else 'red'if x==1 else 'blue' for x in hidden_states[:5000]]
for x, l, c in zip(frame.index[:5000], frame[:5000], color):
    plt.scatter(x, l, alpha=0.8, c=c,marker='.', linewidths=0.01) 
    plt.title('State identification')

fig = plt.figure(figsize=(8,5))
color = ['darkorange' if x>2 else 'green' if x==2 else 'red'if x==1 else 'blue' for x in hidden_states[:5000]]
for x, l, c in zip(tt_diff.index[:5000], tt_diff[:5000], color):
    plt.scatter(x, l, alpha=0.8, c=c,marker='.', linewidths=0.01) 
    plt.title('State identification')
#plt.scatter(gt[:2500].index, gt[:2500], c='red', marker='.', linewidths=0.01, label = 'Gaussian Filtered Series')
#plt.plot(state_prob[:5000]*1000, label = 'State probabilty *1000')
#plt.legend(loc='upper right')
#print(model.fit(np.array(gauss_filt).reshape(-1, 1)).bic(np.array(gauss_filt).reshape(-1, 1)))