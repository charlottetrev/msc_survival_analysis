## Code for regression tree gradient boosting model

## Import packages needed for analysis
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder


## Download data and remove columns
## Adult_dataedit is datafile used for loss = ipcwls for time to event values
## Add absolute path file
adult_data = pd.read_csv('adult_data.csv')
adult_data = adult_data.drop(columns=['Unnamed: 0'])
adult_data = adult_data.drop(columns = ['Number'])
print('got data and removed unnecessary columns...so far so good...')

## Get x and y datasets- separate predictors to y outcome variables
X = adult_data[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity', 'Household_income', 'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke', 'occupation', 'days_active', 'smoking_status']]
y = adult_data[['mortstat', 'permth_int']]

mort = list(y['mortstat'])
time = list(y['permth_int'])
## Change to binary value rather than number
for n,i in enumerate(mort):
    if i == 0:
        mort[n] = False
    else:
        mort[n] = True
mort

## Zip lists together to get list of tuples   
survival = zip(mort, time)
Y = list(survival)

## Need to turn list of tuples into structured array
## Have to tell it what type of data you have in the struct. array
## Get this from the toy data imported above

dt = np.dtype([('fstat', '?'),('lenfol', '<f8')])
Y = np.array(Y,dtype=dt)

## Get test and train data values and then split X data
train_vals, test_vals = train_test_split(range(len(adult_data)), test_size = 0.2)
x_train = X.loc[train_vals].reset_index(drop = True)
x_test = X.loc[test_vals].reset_index(drop = True)

## Get Y outcome data as test and train
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

x_train1 = OneHotEncoder().fit_transform(x_train)
x_test1 = OneHotEncoder().fit_transform(x_test)



# In[ ]:


## Instantiate the GB method

#estimator_gb = GradientBoostingSurvivalAnalysis(n_estimators = 100, learning_rate = 0.1, random_state = 0)

## Import data and removing variables
adult_data2 = pd.read_csv('adult_datatest.csv')
adult_data2 = adult_data2.drop(columns=['Unnamed: 0'])
adult_data2 = adult_data2.drop(columns = ['Number'])
adult_data2

## Get x and y datasets- separate predictors to y outcome variables
X2 = adult_data2[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity', 'Household_income', 'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke', 'occupation', 'days_active', 'smoking_status']]
y2 = adult_data2[['mortstat', 'permth_int']]
X2 = OneHotEncoder().fit_transform(X2)

mort2 = list(y2['mortstat'])
time2 = list(y2['permth_int'])
## Change to binary value rather than number
for n,i in enumerate(mort2):
    if i == 0:
        mort2[n] = False
    else:
        mort2[n] = True
mort2
## Zip lists together to get list of tuples   
survival2 = zip(mort2, time2)
Y2 = list(survival2)

## Need to turn list of tuples into structured array
## Have to tell it what type of data you have in the struct. array
## Get this from the toy data imported above

dt2 = np.dtype([('fstat', '?'),('lenfol', '<f8')])
Y2 = np.array(Y2,dtype=dt2)

## Doing predictions and loops and writing to csv files
## Change headers for whichever datafile is being made
## Loss = ipclws for the time to event values
#with open ('rtcalibrate_timetoevent.csv', 'w', newline = '') as outfile1:
    #writer = csv.writer(outfile1)
    #headers = ['index', 'timetoevent']
    #first = headers
    #writer.writerow(first)
    #res = []
    #estimator_gb = GradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1, max_depth=1, random_state = 0)
    #estimator_gb.fit(x_train1, y_train)

estimator_gb = GradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1, max_depth=1, random_state = 0)
estimator_gb.fit(x_train1, y_train)

## Calibration values for the test set to make plots later
#calibrateres = estimator_gb.predict_survival_function(x_test1)
#calibrateres = pd.DataFrame(data=calibrateres)
#calibrateres.to_csv('rtgb_calibrate.csv')   

### Get the training score for C-statistic
#train_score=estimator_gb.score(x_train1, y_train)
#print('train score for rt gb is: ', train_score)

## Append values from risk scores to open csv code from further up
    #results = estimator_gb.predict(x_test1)
    #for i in results:
        #res.append(i)
        #writer.writerow(res)
        #res = []
## Using new data to produce risk scores and append to csv
#with open ('rtgradboost_newdatariskscores.csv', 'w', newline = '') as outfile1:
#    writer = csv.writer(outfile1)
#    headers = ['index', 'riskscore']
#    first = headers
#    writer.writerow(first)
#    res = []
#    estimator_gb = GradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1, max_depth=1, random_state = 0)
#    estimator_gb.fit(x_train1, y_train)   
#    risks = estimator_gb.predict(X)
#    for i in risks:
#        res.append(i)
#        writer.writerow(res)
#        res = []   

## Get the survival function values to make a plot for the new data
#estimator_gb = GradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1, max_depth=1, random_state = 0)
survfuncs = estimator_gb.predict_survival_function(X2)
## Loop through survival functions and add to a graph
for p in survfuncs:
    plt.step(p.x, p(p.x), where = 'post')
    print(p[1])
plt.ylim(0,1)
plt.ylabel('Survival probability P(T>t) ')
plt.xlabel('Time (months)')
plt.title('RTGB model estimate of survival for 6 test individuals')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('rtgradboosttest.png')
