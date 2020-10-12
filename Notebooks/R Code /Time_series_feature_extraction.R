# Time series feature extraction
library(readr)
library(tsfeatures)
library(gtools)
library(ts)
library(factoextra)
library(ggplot2)
library(rmarkdown)

setwd('~/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data/psykose/patient')
list_names = list.files()
dataset_p <- list()
list_names <- mixedsort(sort(list_names))

for (i in 1:length(list_names)){
  temp_data <- read_csv(list_names[i],  col_names = TRUE ,cols('timestamp' = col_datetime(format = ""), 'date' = col_date(format = ""), 'activity'=col_double()))
  temp_data$date <- NULL
  dataset_p[[i]] <- ts(temp_data$activity,start=1,frequency=24*60)
}

feat_patients <- tsfeatures(dataset_p)

setwd('~/Desktop/Master_Thesis/Schizophrenia_Depression_Project/Data/psykose/control/')
list_names = list.files()
dataset_c <- list()
list_names <- mixedsort(sort(list_names))

for (i in 1:length(list_names)){
  temp_data <- read_csv(list_names[i],  col_names = TRUE ,cols('timestamp' = col_datetime(format = ""), 'date' = col_date(format = ""), 'activity'=col_double()))
  temp_data$date <- NULL
  dataset_c[[i]] <- ts(temp_data$activity,start=1,frequency=24*60)
}
feat_control <- tsfeatures(dataset_c)

feat_control[c('nperiods', 'frequency', 'seasonal_period')] <- NULL
feat_patients[c('nperiods', 'frequency', 'seasonal_period')] <- NULL
feat_patients$condition <- 1
feat_control$condition <- 0

features<- rbind(feat_patients, feat_control)

pat_pca <-prcomp(feat_patients[,c(1:17)], center = TRUE,scale. = TRUE)
p_pca <-data.frame(pat_pca$x[,1],pat_pca$x[,2])
names(p_pca) <- c('PC1','PC2')
ggplot(p_pca, aes(x=PC1, y=PC2)) +geom_point() 

km.res <- kmeans(p_pca, centers = 3, iter.max = 10, nstart = 1)
print(km.res)
p_pca$clusters <- km.res$cluster
ggplot(p_pca, aes(x=PC1, y=PC2, color=clusters)) +geom_point() 

cluster_vec = c(km.res$cluster, replicate(32, 0))


pca  <- prcomp(features[,c(1:17)], center = TRUE,scale. = TRUE)
summary(pca)

new_pca <-data.frame()
new_pca <- cbind(pca$x[,1],pca$x[,2],features[,18] )
new_pca$clusters <-cluster_vec 
names(new_pca) <- c('PC1','PC2','cond','clusters')

ggplot(new_pca, aes(x=PC1, y=PC2, ,color=cond ,shape=clusters)) +geom_point()+scale_shape_identity()


