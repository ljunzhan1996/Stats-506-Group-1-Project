## STATS506 Fall 2018 
## Problem Set 3 Q1
##
## This R script documents the group project on:
## Ridge Regression.
## Dataset: meatspec
##
## Author: Ruirui Zhang, ruiruiz@umich.edu
## Updated: November 10, 2018 - Last modified date

#! Limit lines to 80 characters with rare exceptions. 
# 80: -------------------------------------------------------------------------

# Remove objects from current environment
rm(list = ls())
ls()

# libraries: ------------------------------------------------------------------
# Install Packages ------------------------------------------------------------
#install.packages('dplyr')

# Load Libraries --------------------------------------------------------------
if (!require("dplyr")) {
  install.packages('dplyr')
  library("dplyr")
}

if (!require("MASS")) {
  install.packages('MASS')
  library("MASS")
}

if (!require("glmnet")) {
  install.packages('glmnet')
  library("glmnet")
}

if (!require("ridge")) {
  install.packages('ridge')
  library("ridge")
}

if (!require("ggplot2")) {
  install.packages('ggplot2')
  library("ggplot2")
}

if (!require("caret")) {
  install.packages('caret')
  library("caret")
}

if (!require("faraway")) {
  install.packages('faraway')
  library("faraway")
}
# Sec 3: Data Summary----------------------------------------------------------

# Sec 3.1: Load Data-------------------------------------------------------------
meatspec


# Sec 3.2: Data Exploration ---------------------------------------------------
## Response: fat content
## Predictors: 100 channel spectrum of absorbances
## Number of data points: 215

dim(meatspec) #215, 101
names(meatspec)

# plot first 10 observations
matplot(t(meatspec[c(1:10), c(1:100)]), 
        pch = "*",
        xlab = "channel",
        ylab = "absorbances")

# Sec 3.3: Summarize Data -----------------------------------------------------
summary(meatspec)

# Sec 2: Data Description -----------------------------------------------------

# Sec 2.2: Condition Number ---------------------------------------------------

X = as.matrix(meatspec[, -101])
e = eigen(t(X) %*% X)$val
max(e)/min(e) # 2.6e+14

# Sec 2.1: Correlation  -------------------------------------------------------

par(mar = c(3,3,2,1))
fields::image.plot(cor(X), col = heat.colors(64), las = 1, axes = FALSE)
axis(side = 1, at = seq(0, 100, 10)/100, labels = seq(0, 100, 10))
axis(side = 2, at = seq(0, 100, 10)/100, labels = seq(0, 100, 10), las = 2)
mtext(text = "Channel", side = 1, line = 1, at = 1.1)

# Sec 4: Analysis  ------------------------------------------------------------
# Sec 4.1: Partition Data  ----------------------------------------------------
# method 1: 5 folds, non-equal size
k = 5
flds = createFolds(meatspec, k = k, list = TRUE, returnTrain = FALSE)
names(flds)[1] = "train_data"

# method 2: 10 folds, equal size
n = dim(meatspec)[1]
sample_id = seq(1,n, k)

subset_id = as.list(seq(1,k,1))

select_subset = function(id, data, sample_id){
  #cbind(sample_id+id, sample_id, data[sample_id+id,])
  select_id = sample_id + id 
  if(max(select_id) > n) select_id = select_id[-length(sample_id)]
  data[select_id,]
}

subset_data = lapply(X = subset_id, FUN =select_subset, meatspec, sample_id )
# length(subset_data[[1]][,101])

# method 2a: 10 folds, equal size
new_data = meatspec %>%
  dplyr::mutate(gp = seq(1:n())%% 5)


# Sec 4: Analysis - glmnet
# Sec 4.0: select a sequence of lambda  -------------------------------
fit_all = glmnet(x = as.matrix(new_data[which(new_data$gp != 0),1:100]),
                 y = as.matrix(new_data[which(new_data$gp != 0),101]),
                 alpha = 1)
lambda_seq = fit_all$lambda 

# Sec 4.1: Ridge Regression for subset 1:(k-1)  -------------------------------

for (i in 1:(k-1)){
  
  temp_data = new_data[which(new_data$gp == i), ]
  assign(paste("fit", i, sep = ""),
         glmnet(x = as.matrix(temp_data[ ,1:100]),
                y = as.matrix(temp_data[ ,101]),
                lambda = lambda_seq)
         )
}


# Sec 4.2: calculate mse  -------------------------------
# cycle through folds, using each in turn as the validation dataset,
# other for training 

rmse_fct = function(y, yhat){
  sqrt(mean( (y - yhat)^2))
}

mse_list = list()
lambda_list = list()
for (i in 1:(k-1)){

  assign("fit", get(paste("fit",i, sep = "")))
  
  validation_data = new_data[which(new_data$gp != i & new_data$gp != 0), ]
  
  y = validation_data[,101]
  
  yhat = coef(fit)[1] +
    as.matrix(validation_data[,1:100]) %*% 
    coef(fit)[2:ncol(subset_data[[i]]),]
  
  mse_list[[i]] = sqrt(colMeans((y - yhat)^2))
  
  # lambda_list[[i]] = lambda_seq[which.min(mse_list[[i]])]
}

mse_list
# lambda_list

# Sec 4.3: select lambda  -------------------------------
mse_matrix = matrix(unlist(mse_list), 
                    nrow = length(mse_list[[1]]),
                    ncol = length(mse_list)) 
mse_mean = apply(X = mse_matrix, MARGIN = 1, FUN = mean)


lambda_fit = lambda_seq[which.min(mse_mean)]


# Sec 4.3: prediction  -------------------------------

fit_final = glmnet(x = as.matrix(new_data[which(new_data$gp != 0),1:100]),
                   y = as.matrix(new_data[which(new_data$gp != 0),101]),
                   alpha = 1,
                   lambda = lambda_fit)

coef(fit_final)[which(coef(fit_final)!= 0),]

pred_y = predict(object = fit_final,
                 newx = as.matrix(new_data[which(new_data$gp == 0), 1:100]))


mse_fct(y = new_data[which(new_data$gp == 0), 101],
        yhat = pred_y)

cbind(real_y = new_data[which(new_data$gp == 0), 101], 
      glmnet_y = pred_y)

summary(new_data[which(new_data$gp == 0), 101] - pred_y)






# Sec 5: Analysis: Ridge Regression using package RIDGE  -------------------------------
# Sec 5.0: Ridge regression  -------------------------------
ridge.fit.error = linearRidge(formula = fat ~ ., 
                              data = new_data[which(new_data$gp != 0),1:101],
                              lambda = "automatic",
                              nPCs = NULL,)

# Sec 5.1: Ridge regression - nPCs = 0 -------------------------------

ridge.fit = linearRidge(formula = fat ~ ., 
                        data = new_data[which(new_data$gp != 0),1:101],
                        nPCs = 1)
ridge.fit_summary = summary(ridge.fit)
ridge.fit_summary$lambda
coef(ridge.fit)
plot(ridge.fit)
print(ridge.fit)


# Sec 5.2: Prediction  -------------------------------
ridge.predict = predict(object = ridge.fit, newdata = new_data[which(new_data$gp == 0), 1:100])

ridge.predict_cal = coef(ridge.fit)[1] + 
                rowSums(as.matrix(new_data[which(new_data$gp == 0), 1:100]) %*% coef(ridge.fit)[2:101])
true_y = new_data[which(new_data$gp == 0),101]

rbind(ridge.predict_cal,
      ridge.predict,
      true_y)

mse_fct(y = new_data[which(new_data$gp == 0), 101],
        yhat = ridge.predict)
