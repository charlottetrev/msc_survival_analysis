##plots of time to event against predicted time to event
## Get relevant libraries
library(ggplot2)
library(tidyverse)
library(dplyr)
library(survival)
library(gbm)

library(rpart)
data(kyphosis)
y <- as.numeric(kyphosis$Kyphosis)-1
x <- kyphosis$Age
glm1 <- glm(y~poly(x,2),family=binomial)
p <- predict(glm1,type="response")
calibrate.plot(y, p, xlim=c(0,0.6), ylim=c(0,0.6))




##Download test data to get tables
test_data <- read.csv('test_data.csv')

###CALIBRATION DATA PLOTS- OBS VS PRED VALUES/RISK SCORES
## Open relevant datasets for time to event data
## Add new column to test data to use in plots

tte_cw <- read.csv('cwgradboost_timetoevent.csv')
riskcw <- tte_cw$index
test_data$cwriskscore = riskcw

tte_rt <- read.csv('rtgradboost_timetoevent.csv')
riskrt <- tte_rt$index
test_data$rtriskscore = riskrt

rsf_risk <- read.csv('rsfriskscores_results.csv')
rsfriskscore <- rsf_risk$riskscore
test_data$rsfrisk = rsfriskscore

##change plots to only show those who died in obs vs pred comparison
test_data_plot <- test_data %>% filter(mortstat == 1)
## Make plots for survival times vs survival time
cwplot <- ggplot(test_data_plot, aes(x=cwriskscore, y=permth_int, color = Gender)) + geom_point() + labs(title='CWGB observed and predicted time to event values', y='Observed time to event(months)', x='Predicted time to event(months)') + geom_abline() + xlim(0,400) + ylim(0,400)
cwplot
cwplot + facet_wrap( ~Gender, ncol=2)

rtplot <- ggplot(test_data_plot, aes(x=rtriskscore, y=permth_int, color = Gender)) + geom_point() + labs(title='RTGB observed and predicted time to event values', y='Observed time to event(months)', x='Predicted time to event(months)') + geom_abline() + xlim(0,300) + ylim(0,300)
rtplot
rtplot + facet_wrap( ~Gender, ncol=2)

rsfplot <- ggplot(test_data_plot, aes(x=rsfrisk, y=permth_int, color = Gender)) + geom_point() + labs(title='RSF observed and predicted time to event values', y='Observed time to event(months)', x='Predicted time to event(months)') + geom_abline() + xlim(0,300) + ylim(0,300)
rsfplot
rsfplot + facet_wrap( ~Gender, ncol=2)


###CALIBRATION PLOTS WITH SURVIVAL FUNCTION VALUES
## Get the 10 year predictions from the datasets
## Add to the test_data to use in the plots
rsf_calibrate <- read.csv('rsfcalibrate.csv')
rsf_calib <- rsf_calibrate$X120
test_data$rsf_calib = rsf_calib
test_data$rsf_calib <- 1 - test_data$rsf_calib 

calibrate.plot(test_data$mortstat, test_data$rsf_calib, xlab ='Predicted survival value', ylab='Observed average survival time', main ='Calibration plot of RSF')


rtgb_calibrate <- read.csv('rtgb_calibrate.csv')
rtgb_calib <- rtgb_calibrate$X120
test_data$rtgb_calib = rtgb_calib
test_data$rtgb_calib <- 1 - test_data$rtgb_calib
calibrate.plot(test_data$mortstat, test_data$rtgb_calib, xlab ='Predicted survival value', ylab='Observed average survival time', main ='Calibration plot of Regression Tree GB')

cwgb_calibrate <- read.csv('cwgb_calibrate.csv')
cwgb_calib <- cwgb_calibrate$X120
test_data$cwgb_calib = cwgb_calib
test_data$cwgb_calib <- 1 - test_data$cwgb_calib
calibrate.plot(test_data$mortstat, test_data$cwgb_calib, xlab ='Predicted survival value', ylab='Observed average survival time', main ='Calibration plot of Component-Wise GB', xlim=c(0,0.9), ylim=c(0,0.9))