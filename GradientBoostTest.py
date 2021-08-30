### Code for Component-Wise Gradient Boosting model

## Import necessary packages
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder



##adultdataedit is datafile used for loss = ipcwls in gradient booster for...
## time to event data

## Add absolute path file to open data
adult_data = pd.read_csv('allimputed.csv')
#adult_data = adult_data.drop(columns=['Unnamed: 0'])
adult_data = adult_data.drop(columns = ['Number'])
print('got data and removed unnecessary columns...so far so good...')

## Get x and y datasets- separate predictors to y outcome variables
X = adult_data[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity', 'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke','days_active', 'smoking_status']]
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

## Need to turn list of tuples into structured array
## Have to tell it what type of data you have in the struct. array
## Get this from the toy data imported above

dt = np.dtype([('fstat', '?'),('lenfol', '<f8')])
Y = np.array(Y,dtype=dt)

## get test and train data values and then split X data into test and train
train_vals, test_vals = train_test_split(range(len(adult_data)), test_size = 0.2, random_state=1)
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

x_train1 = OneHotEncoder().fit_transform(x_train)
x_test1 = OneHotEncoder().fit_transform(x_test)


## Instantiate the GB method

estimator_cwgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1, random_state = 0)


## TEST DATA
## Import data and removing variables for new dataset to test
adult_data2 = pd.read_csv('adult_datatest2.csv')
#adult_data2 = adult_data2.drop(columns=['Unnamed: 0'])
adult_data2 = adult_data2.drop(columns = ['Number'])
adult_data2

## Get x and y datasets- separate predictors to y outcome variables
X2 = adult_data2[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity', 'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke','days_active', 'smoking_status']]
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

## Write to csv file with appropriate headers depending on the task being done 
#with open ('cwgradboost_coefs.csv', 'w', newline = '') as outfile1:
    #writer = csv.writer(outfile1)
    #headers = ['model_num', 'cstatistic', 'nestimators']
    #headers = ['index', 'riskscore']
    #first = headers
    #writer.writerow(first)
    #model_num = 1
    #res = []
    #estimator_cwgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1, random_state = 0, loss = 'ipcwls')
    #estimator_cwgb.fit(x_train1, y_train)
    #risks = estimator_cwgb.predict(x_test1)
    #risks2 = estimator_cwgb.predict(X2)
    #coefs = pd.Series(risks.coef_, ['Intercept'] + x_test1.columns.tolist())
    #for coef in coefs:
        #res.append(coef)
        #writer.writerow(res)

## Get values for calibration plots
#estimator_cwgb.fit(x_train1, y_train)
#test_score = estimator_cwgb.score(x_test1, y_test)
#print('test score = ', test_score)
#calibrateres = estimator_cwgb.predict_survival_function(x_test1)
#calibrateres = pd.DataFrame(data=calibrateres)
#calibrateres.to_csv('cwgb_calibrate.csv')
#print('made csv for results')


## Get the train score for the c-statistic
#train_score = estimator_cwgb.score(x_train1, y_train)
#print('train score for cwgb is: ', train_score)

## Write the risks data to a csv file to get risk scores and time to event...
## Fits the open csv command above

    #for i in risks:
        #res.append(i)
        #writer.writerow(res)
        #res = []
    #breaks = ['BREAK', 'BREAK']
    #writer.writerow(breaks)
    #risks2 = estimator_cwgb.predict(X)
    #for j in risks2:
        #res.append(j)
        #writer.writerow(res)
        #res = []
## Estimate the risk scores for the test set
    #cstat = estimator_cwgb.score(x_test1, y_test)
    #writer.writerow(res)
    #model_num += 1
    #print(model_num)
    #res = []

## Open new dataset and calculate risk scores and add to csv
with open ('cwtimetoevent_.csv', 'w', newline = '') as outfile1:
    writer = csv.writer(outfile1)
    headers = ['index', 'riskscore']
    first = headers
    writer.writerow(first)
    res = []
    estimator_gb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1,random_state = 0, loss = 'ipcwls')
    estimator_gb.fit(x_train1, y_train)   
    risks = estimator_gb.predict(x_test1)
    for i in risks:
        res.append(i)
        writer.writerow(res)
        res = []   


## Way to get survival function predictions for new data x6
#estimator_cwgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators = 150, learning_rate = 0.1, random_state = 0)
#estimator_cwgb.fit(x_train1, y_train)
#survfuncs = estimator_cwgb.predict_survival_function(X2)

## Loop through predictions and plot on a graph
#for p in survfuncs:
    #plt.step(p.x, p(p.x), where = 'post')
    #print(1)
#plt.ylim(0,1)
#plt.ylabel('Survival probability P(T>t)')
plt.xlabel('Time (months)')
#plt.title('CWGB model estimate of survival for 6 test individuals')
#plt.legend()
#plt.grid(True)
#plt.show()
#plt.savefig('cwgradboosttest.png')
