## Import necessary libraries

library(survival)
library(ggplot2)
library(tidyverse)
library(glmnet)

## 
adult_data <- load('C:/Users/charl/OneDrive/Desktop/Data Science/Semester 2/Dissertation/Code/Model Code/adult_nhim.RData')

## NEW APPROACH: All Possible Models
survival.outcome <- Surv(adult_data$permth_int, adult_data$mortstat)

#vectors for event/vars
outcome <- c('survival.outcome')
predictors <- c(names(adult_data[2:8]))
dataset <- adult_data


#create models to run through 

model_list <- lapply(seq_along((predictors)), function(n) {
  
  left <- outcome
  right <- apply(X = combn(predictors, n), MARGIN = 2, paste, collapse = ' + ')
  paste(left, right, sep = ' ~ ')
  
})

model_vec <- unlist(model_list)

#fit models

model_fits <- lapply(model_vec, function(x) {
  formula <- as.formula(x)
  fit <- coxph(formula, data = dataset)
  result.AIC <- extractAIC(fit)
  
  data.frame(AIC = result.AIC[2],
             model = x)
})

#put into dataframe
result <- do.call(rbind, model_fits)

#sort dataframe
library(doBy)
final_vs <- orderBy(~ AIC, result)

save(final_vs, file = 'final_vs.RData')

write.csv(final_vs, file = 'final_vs.csv')



## Automated model development?

#varlist <- names(adult_data)[2:19]

#purrr::map(varlist, ~coxph(as.formula(paste("Surv(adult_data$permth_int, adult_data$mortstat) ~  Age + ", .x)), adult_data))
##this makes models for each variable + Age
