
# Analysis of circadian and ultra-circadian cycles to improve data augmentation techniques for time series for classification of schizophrenia patients

## Problem statement
Medical data is known for small sample sizes and thus methods can lack generalization. 
The given Psykose dataset contains time series data of 55 individuals, 23 of these individuals suffer a medical condition while 32 individuals belong to the control group. The Psykose dataset measured actigraphy time series data of schizophrenic individuals. The time series per individual have a sample size up to 21500 samples, which corresponds to a measurement of wrist acceleration by minute over two weeks.   

## Project idea
to improve the classification of actigraph time series is to segment each time series in active and resting phases. The extracted segments are then used as input for the classification. 
Hidden Markov Models are used to segment the time series period in the two different phases. The segmentation procedure. The decoding problem identifies the underlying hidden states which generate the observed process. 
In this case, the hidden states are assumed to be the resting/ active periods. 
A heuristic to determine the starting solution will be proposed and evaluated. 
For each identified active and resting period, time series features from fitted models will be extracted. Moreover, feature describing the relation between different periods are introduced. 
A SVM or logistic regression will be used as a classification algorithm for the feature space. 
Additional, ways of augmenting the different periods per patient can be used to feed a LSTM Network for an improved classification. 


## Process
1. Explorative Analysis of the Actigraphy Time Series
2. Analysis of circadian and ultra-circadian cycles in the frequency domain
3. Hidden Markov Models for Time Series Segmentation
	3.1 State Estimation
	3.2 Parameter Estimation 
5. Statistical Modeling of extracted Time Series Segments  
6. Classification based on statistical Features 

## Literature

## Papers datasets
1. https://dl.acm.org/doi/pdf/10.1145/3204949.3208125
2. https://osf.io/e2tzf

## Data
1. https://datasets.simula.no/depresjon/
2. https://datasets.simula.no/psykose/
