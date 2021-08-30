##Missing data imputation
adult_nh <- read.csv('adult_nh.csv')
## Download relevant libraries
library(mice)

## Identify how many values are NA across the columns...
## ...produces tables with NA frequency
numberna <- map(adult_nh, ~sum(is.na(.)))

## Collect summary information about the dataset
before_impute <- summary(adult_nh)

## Impute data giving dataset and number of times ...
## you want the imputation to be done (m=1)
adult_nhi <- mice(adult_nh, m=1)

adult_nhim <- complete(adult_nhi, 1)
adult_nhim

## Checking the number of NAs after imputation...
## (checking imputation worked)
numna <- map(adult_nhim, ~sum(is.na(.)))

after_imput <- summary(adult_nhim)
after_imput
before_impute

## Remove unnecessary columns
adult_nhim <- adult_nhim[-c(24,13)]
adult_nhim

## Save the file to be used again
save(adult_nhim, file = 'adult_nhim.RData')

write.csv(adult_nhim, file = 'adult_nhim.csv')
glimpse(adult_nhim)

## Open dataset and only select relevant columns
adult_data <- adult_nhim[-c(3, 8, 13, 20, 22, 25, 26, 27, 28, 29, 31, 32, 33, 35)]
adult_data

## Save new dataset with irrelevant columns removed
save(adult_data, file = 'adult_data.RData')
write.csv(adult_data, file = 'adult_data.csv')


alldata <- read.csv('adult_data.csv')
alldata <- alldata[3:22]
alldata <- select(alldata, -c(occupation, Household_income))
## changing values to make 2 groups in variables 
alldata$smoking_status[alldata$smoking_status == 2] <- 1
alldata$smoking_status[alldata$smoking_status ==3] <- 2
alldata$heart_attack[alldata$heart_attack==9] <- NA
alldata$heart_attack[alldata$heart_attack==7] <- NA
alldata$relative_ha[alldata$relative_ha==9] <- NA
alldata$liver_problem[alldata$liver_problem==9] <- NA
alldata$cancer[alldata$cancer==9] <- NA
alldata$stroke[alldata$stroke==9] <- NA

numberna <- map(alldata, ~sum(is.na(.)))
numberna
alldataim <- mice(alldata, m=1)
alldataim <- complete(alldataim, 1)
alldataim
## scale the BP and cholesterol variables
alldataim$Systolic = alldataim$Systolic/10
alldataim$Diastolic = alldataim$Diastolic/10
alldataim$Chol2 = alldataim$Chol2/10
write.csv(alldataim, file='allimputed.csv')

alldata2 <-dummy_cols(alldataim, select_columns =c('Ethnicity'), remove_selected_columns = TRUE)
save(alldata2, file = 'adult_data2.RData')
write.csv(alldata2, file = 'adult_data2.csv')

