## importing packages
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt

## Import data and removing variables
adult_data = pd.read_csv('adult_data.csv')
adult_data = adult_data.drop(columns=['Unnamed: 0'])
adult_data = adult_data.drop(columns = ['Number'])
adult_data

## Get x and y datasets- separate predictors to y outcome variables
X = adult_data[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity', 'Household_income', 'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke', 'kidney_disease', 'occupation', 'days_active', 'smoking_status']]
y = adult_data[['mortstat', 'permth_int']]

mort = list(y['mortstat'])
time = list(y['permth_int'])
## change to binary value rather than number
for n,i in enumerate(mort):
    if i == 0:
        mort[n] = False
    else:
        mort[n] = True
mort
## zip lists together to get list of tuples   
survival = zip(mort, time)
Y = list(survival)

## need to turn list of tuples into structured array
## have to tell it what type of data you have in the struct. array
## get this from the toy data imported above

dt = np.dtype([('fstat', '?'),('lenfol', '<f8')])
Y = np.array(Y,dtype=dt)

## get test and train data values and then split X data
train_vals, test_vals = train_test_split(range(len(adult_data)), test_size = 0.2)
x_train = X.loc[train_vals].reset_index(drop = True)
x_test = X.loc[test_vals].reset_index(drop = True)

## get Y outcome data as test and train
y_train = []
for val in train_vals:
    y_train.append(Y[val])
y_train = np.asarray(y_train)
#print(y_train)

y_test = []
for val in test_vals:
    y_test.append(Y[val])
y_test = np.asarray(y_test)
#print(y_test)
print('starting to print to csv')

## open file and set up headers for model
with open ('rsfmodel6_results.csv', 'w', newline = '') as outfile1:
    writer = csv.writer(outfile1)
    headers = ['model_number', 'num_trees', 'split', 'leaf', 'c_statistic']
    first = headers
    writer.writerow(first)
    
    model_values = []
    model_num = 0
    for t in (100, 150, 175):
        for s in (10, 50, 75):
            for l in (10, 50, 75):
                rsf = RandomSurvivalForest(n_estimators= t,
                                           min_samples_split= s,
                                           min_samples_leaf= l,
                                           max_features= 6,
                                           n_jobs=-1,
                                           random_state= None)
                estimate = rsf.fit(x_train, y_train)
                pred = rsf.predict_survival_function(x_test)
                cstat = rsf.score(x_test, y_test)
                model_values.append(model_num)
                model_values.append(t)
                model_values.append(s)
                model_values.append(l)
                model_values.append(cstat)
                writer.writerow(model_values)
                model_values = []
                model_num +=1
                print(model_num)