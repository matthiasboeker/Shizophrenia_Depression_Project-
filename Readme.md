
# Analysis of circadian and ultra-circadian cycles to classify schizophrenic and non-schizophrenic persons


## Process
1. Explorative analysis of the time series
2. Features extraction
3. Basic logistic regression

## Explorative Analysis
* Extract the daily(24h/12h) and nightly structure
* Visualize autocorrelation, seasonality, boxplots, Fourier series
* Fourier transformation
## Extract features
    * Daily/ Nightly mean  
    * Daily/ Nightly variance
    * Zero proportion
    * Day and night differences
    * Daily differences
    * Fourier amplitude and phase
## Basic logistic regression model
* Check for multicollineartiy
* Check for non-linearity


## Papers
1. https://dl.acm.org/doi/pdf/10.1145/3204949.3208125
2. https://osf.io/e2tzf

## Data
1. https://datasets.simula.no/depresjon/
2. https://datasets.simula.no/psykose/
