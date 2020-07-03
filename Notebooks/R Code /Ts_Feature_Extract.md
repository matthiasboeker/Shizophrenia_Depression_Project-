Load in Data
------------

The dataset contains actigraphy time series of 22 schizophrenic patients and 32 control persons. Actigraph data is provided by a wrist advice which calculates the acceleration. The dataset will be investigated in order to to improve classification of from Schizophrenia suffering persons.

``` r
setwd('~/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data/psykose/patient')
list_names = list.files()
dataset_p <- list()
list_names <- mixedsort(sort(list_names))
for (i in 1:length(list_names)){
  temp_data <- read_csv(list_names[i],  col_names = TRUE ,cols('timestamp' = col_datetime(format = ""), 'date' = col_date(format = ""), 'activity'=col_double()))
  temp_data$date <- NULL
  dataset_p[[i]] <- ts(temp_data$activity,start=1,frequency=24*60)
}
setwd('~/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data/psykose/control/')
list_names = list.files()
dataset_c <- list()
list_names <- mixedsort(sort(list_names))

for (i in 1:length(list_names)){
  temp_data <- read_csv(list_names[i],  col_names = TRUE ,cols('timestamp' = col_datetime(format = ""), 'date' = col_date(format = ""), 'activity'=col_double()))
  temp_data$date <- NULL
  dataset_c[[i]] <- ts(temp_data$activity,start=1,frequency=24*60)
}
setwd('~/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Notebooks')
```

Extract Time series features
----------------------------

Rob Hyndman provides a R package which extract relevant time series features like: \* Stability \* Autocorrelation coefficients \* Unit roots

In total 27 features were extracted. More details can be found: <http://business.monash.edu/econometrics-and-business-statistics/research/publications>

    ## # A tibble: 6 x 18
    ##    trend   spike linearity curvature e_acf1 e_acf10 seasonal_streng…  peak
    ##    <dbl>   <dbl>     <dbl>     <dbl>  <dbl>   <dbl>            <dbl> <dbl>
    ## 1 0.102  5.01e-9    -15.6      -19.5  0.708   1.83             0.296   210
    ## 2 0.0508 3.86e-8      7.58     -10.0  0.528   0.652            0.194   854
    ## 3 0.0862 9.26e-9    -15.9      -19.3  0.432   0.647            0.281   667
    ## 4 0.0745 4.95e-9     -8.68     -18.6  0.628   1.39             0.454   166
    ## 5 0.0884 8.08e-9     -6.69     -24.5  0.624   1.35             0.376   401
    ## 6 0.0457 1.45e-8     -6.45     -14.4  0.567   0.759            0.311   625
    ## # … with 10 more variables: trough <dbl>, entropy <dbl>, x_acf1 <dbl>,
    ## #   x_acf10 <dbl>, diff1_acf1 <dbl>, diff1_acf10 <dbl>, diff2_acf1 <dbl>,
    ## #   diff2_acf10 <dbl>, seas_acf1 <dbl>, condition <dbl>

A PCA is applied to reduce the feature space down to 2 Eigenvectors. The Eigenvectors are scatter to identify potential clusters/ differences with in the two groups (patient/control).

![](Ts_Feature_Extract_files/figure-markdown_github/pca-1.png)

Points annoted with a 1 are Schizophrenic patients (light blue), other are annoted with a 0. We can observe, that the upper left part seems to build a cluster but there are some Patient points which clearly fall in the same field as the control group. It can be assumed that there might be different clusters within the Patients data.

A PCA to only the Patients data will be conducted to analyse differences and visualize clusters.

![](Ts_Feature_Extract_files/figure-markdown_github/pca_to_patients-1.png)

Conducting a kmean cluster analysis works out three clear clusters within the patients data. The cluster information of the patients data will now be transferred into the overall scatterplot of patients and the control group.

![](Ts_Feature_Extract_files/figure-markdown_github/overall_cluster:analysis-1.png)

As we see in the scatter plot, one of the cluster we worked out within the patients dataset is indeed the one falling in within the field of the control. Further analysis into the acutal features as to be conducted to work out the actual difference in the time series features of the Patients and the control group.
