
#Analysis of circadian and ultra-circadian cycles to improve data augmentation techniques for time series for classification of schizophrenia patients

## Problem statement
Medical data is known for small sample sizes and thus methods can lack generalization. In the given data set of Psykose, there is a small sample size given the number of patients but a highly sampled time series per patient.
Thus, the large amount of data within a single time series should be used for data augmentation to enrich the small sample size of patients. Moreover, the information about circadian cycles and ultra circadian cycles should be preserved while applying augmentation methods.

## Project idea
The idea is to frame the time series and it's underlying pattern with dynamic models like ARIMA, GARCH, Hidden-Markov Models and/or frequency models like Fourier decompositions. Since the estimated parameters for a time series are a one sample for the patients dataset, a parameter distribution can be approximated (bootstrapping). From this approximated parameter distribution parameter values will be randomly drawn to create synthetic time series.

## Process
1. Explorative analysis of the time series
2. Analysis of circadian and ultra-circadian cycles in the frequency domain
2. Comparative study of statistical models
3. Approximation of parameter distribution
4. Creation of synthetic time series (validation?)
5. Implement baseline ML classifier for comparative experiment
6. Implement CNN with augmented spectrogram pictures as comparative

## Literature
1. 

## Papers datasets
1. https://dl.acm.org/doi/pdf/10.1145/3204949.3208125
2. https://osf.io/e2tzf

## Data
1. https://datasets.simula.no/depresjon/
2. https://datasets.simula.no/psykose/
