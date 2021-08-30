library(survival)
library(ggplot2)
library(tidyverse)
library(knitr)
library(survminer)
library(rms)
library(dplyr)
library(broom)
library(ggfortify)
library(fastDummies)
library(mice)


## Open data in python to get test and train dataset splits


## Look at correlation of variables and can remove any with very high correlation
#cor(adult_nhim)

## Data edit is the one with 0.1 added to 0s
test_data2 <- read.csv('adult_dataedit.csv')
## Test data is the 1122 length dataset
test_data <- read.csv('testnew_data.csv')
test_data <-dummy_cols(test_data, select_columns =c('Ethnicity'), remove_selected_columns = TRUE)
test_data <- test_data[3:26]
## Adultdatatest is the 6 people test
test_data6 <- read.csv('adult_datatest.csv')
test_data6 <- test_data6[3:24]

## Got a section of training data from python to check training
## To run the data analysis with the training data instead...
## to check the training scores, change the name of the dataset...
## within the coxph objects when fitting the models
train_data <- read.csv('train_data.csv')
train_data <- train_data[3:26]


## CPH Model 1- full model

cox.mod5 <- coxph(Surv(time =test_data$permth_int, 
                       event = test_data$mortstat) ~ BMI + 
                    Systolic+ Diastolic+ regularity + 
                    Ethnicity_2+ Ethnicity_3 + Ethnicity_4+ Ethnicity_5+
                   Gender + Age + Chol2 + cancer + stroke +
                    relative_ha + heart_attack + liver_problem +
               days_active + smoking_status, data = test_data)

summary(cox.mod5)
## Look into proportional hazards assumptions and make table
prophaz1 <- cox.zph(cox.mod5)
prophaz1
print(prophaz1)
## Make plot of the proportional hazards assumption...
## Plots a graph for each variable
plot(prophaz1)
cox4 <- survfit(coxph(Surv(time = test_data$permth_int, event = test_data$mortstat) ~ BMI + Systolic+ Diastolic+ regularity+ Ethnicity_2 + Ethnicity_3 + Ethnicity_4+ Ethnicity_5+ Gender + Age + Chol2 + heart_attack+ relative_ha+ liver_problem + cancer + stroke + days_active + smoking_status, data=test_data), newdata=test_data6)
cox4 <- survfit(cox.mod4, test_data)
cox4

##survival function plot- essentially plots a life table?
res <- summary(cox4)
cols <- lapply(c(1:6) , function(x) res[x])
lifetable1 <- do.call(data.frame, cols)
head(lifetable1)
save(lifetable1, file='survfunccph21.Rdata')



plot(cox4, ylim=c(0, 1), xlab='Time (months)', ylab='Survival probability P(T>t)', main='CPH model 1 estimate of survival ', col=c(1,2,3,4,5,6))


## Make hazard ratio plot
ggforest(cox.mod5, data = test_data, main ='Hazard ratio of coefficients for CPH 1')
## Get median survival time
#median <- survfit(Surv(time=test_data2$permth_int, event=test_data2$mortstat) ~ 1, data = adult_data)
#median

## Summary info about the models
summary(cox.mod5)
cox5aic <- AIC(cox.mod5)
cox5aic
concordance(cox.mod5)[1]

##Test calibration of the model- only works with the edited dataset..
x <- cph(Surv(time = test_data$permth_int, event = test_data$mortstat) ~ BMI + Systolic+ Diastolic+ regularity + Ethnicity_2 + Ethnicity_3 +Ethnicity_4 +Ethnicity_5+ Gender + Age + Chol2 + heart_attack+ relative_ha+ liver_problem + cancer + stroke + days_active + smoking_status, x=TRUE, y=TRUE, surv=TRUE, time.inc=120, dxy=TRUE, data = test_data)
x1 <- calibrate(x, u=120)
plot(x1, main = 'Calibration plot for Cox PH model 1', xlab='Predicted 120 Month Survival', ylab='Fraction Surviving 120 Months')


## CPH2- all possible model of variable selection
## Make model using test data
cox.mod01 <- coxph(formula = Surv(time =test_data$permth_int,
             event = test_data$mortstat) ~ Systolic + Diastolic +
               regularity + Gender + Age + relative_ha + cancer +
               stroke + days_active + smoking_status,
             data = test_data)
## Summary information of the models
summary(cox.mod01)
## Assess proportional hazards assumptions by making plots of variables
prophaz01 <- cox.zph(cox.mod01)
prophaz01
print(prophaz01)
plot(prophaz01)
## Check AIC of the model
cox01aic <- AIC(cox.mod01)
cox01aic <- as.double(cox01aic)
cox01aic

#anova(cox.mod4, cox.mod01)
## Check concordance of the model
concordance(cox.mod01)

## Check calibration of the model- create a survfit object
cox5 <- survfit(coxph(Surv(time = test_data$permth_int, event = test_data$mortstat) ~ Systolic + Diastolic + regularity + Gender + Age + relative_ha + cancer + stroke+ days_active + smoking_status, data = test_data),newdata=test_data6)

res2 <- summary(cox5)
cols <- lapply(c(1:6) , function(x) res[x])
lifetable2 <- do.call(data.frame, cols)
head(lifetable2)
save(lifetable2, file='survfunccph22.Rdata')



plot(cox5, ylim=c(0, 1), xlab='Time (months)', ylab='Survival probability P(T>t)', main='CPH model 2 estimate of survival ', col=c(1,2,3,4,5,6))


#plot(cox01, ylim=c(0.85, 1), xlab='Months', ylab='Proportion Alive', main = 'Survival Function for 2nd Cox Model')
x2 <- cph(Surv(time = test_data$permth_int, event = test_data$mortstat) ~ Systolic + Diastolic + regularity + Gender + Age + relative_ha + cancer + stroke + days_active + smoking_status, x=TRUE, y=TRUE, surv=TRUE, time.inc=120, dxy=TRUE, data = test_data)
x2 <- calibrate(x2, u=120)
plot(x2, main = 'Calibration plot for Cox PH model 2', xlab='Predicted 120 Month Survival', ylab='Fraction Surviving 120 Months')

## Make hazard ratio plots for the second model
ggforest(cox.mod01, data = test_data, main ='Hazard ratio of coefficients for CPH 2')