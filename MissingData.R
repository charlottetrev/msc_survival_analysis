##Missing data imputation

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

