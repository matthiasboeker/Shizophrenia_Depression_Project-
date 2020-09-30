#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:06:10 2020

@author: matthiasboeker
Basic Feature Extraction of the wake periods  
"""

#Extract the mean, standard deviation 
series_p = range(0,len(shizophrenia_p))
series_c = range(0,len(shizophrenia_c))
feat_mean_p = np.concatenate([[np.mean(patients_wake_periods[n][k]) for k in range(0,len(patients_wake_periods[n]))] for n in series_p]).ravel()
feat_mean_c = np.concatenate([[np.mean(control_wake_periods[n][k]) for k in range(0,len(control_wake_periods[n]))] for n in series_c]).ravel()
feat_std_p = np.concatenate([[np.std(patients_wake_periods[n][k]) for k in range(0,len(patients_wake_periods[n]))] for n in series_p]).ravel()
feat_std_c = np.concatenate([[np.std(control_wake_periods[n][k]) for k in range(0,len(control_wake_periods[n]))] for n in series_c]).ravel()
feat_len_p = np.concatenate([[len(patients_wake_periods[n][k]) for k in range(0,len(patients_wake_periods[n]))] for n in series_p]).ravel()
feat_len_c = np.concatenate([[len(control_wake_periods[n][k]) for k in range(0,len(control_wake_periods[n]))] for n in series_c]).ravel()
feat_zeros_p = np.concatenate([[len(patients_wake_periods[n][k].loc[patients_wake_periods[n][k]==0])/len(patients_wake_periods[n][k]) for k in range(0,len(patients_wake_periods[n]))] for n in series_p]).ravel()
feat_zeros_c = np.concatenate([[len(control_wake_periods[n][k].loc[control_wake_periods[n][k]==0])/len(control_wake_periods[n][k]) for k in range(0,len(control_wake_periods[n]))] for n in series_c]).ravel()




feat_p  = {'mean': feat_mean_p, 'std': feat_std_p,'f.prop':feat_zeros_p ,'length':  feat_len_p,'label': np.repeat(1, len(feat_std_p))}
feat_df_p = pd.DataFrame(data=feat_p)
feat_c  = {'mean': feat_mean_c, 'std': feat_std_c, 'f.prop':feat_zeros_c,'length':  feat_len_c,'label': np.repeat(0, len(feat_std_c))}
feat_df_c = pd.DataFrame(data=feat_c)
feat = pd.concat([feat_df_p, feat_df_c], axis=0, sort=False)
feat = feat.reset_index()

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
X = feat.drop(['label','index'], axis=1)
y = feat['label']

fig = plt.figure(figsize=(8,5))
color = ['darkorange' if x>0 else 'navy' for x in y]
shape = ['d' if k>0 else 'd' for k in y]
for x, l, c, m in zip(feat['f.prop'], feat['std'], color, shape):
    plt.scatter(x, l, alpha=0.8, c=c,marker=m)  
    plt.title('Features')


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns= X.columns.values)
# pca of features spaces 
pca = KernelPCA(n_components=4, kernel='linear')
principalComponents_all = pca.fit_transform(X)


fig = plt.figure(figsize=(8,5))
color = ['darkorange' if x>0 else 'navy' for x in y]
shape = ['d' if k>0 else 'd' for k in y]
for x, l, c, m in zip(principalComponents_all[:,0], principalComponents_all[:,1], color, shape):
    plt.scatter(x, l, alpha=0.8, c=c,marker=m)  
    plt.title('Features')
