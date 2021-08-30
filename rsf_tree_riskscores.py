## Running random forests

## Importing packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
import csv
from sksurv.preprocessing import OneHotEncoder

## Import data and removing variables...
adult_data = pd.read_csv('allimputed.csv')
#adult_data = adult_data.drop(columns=['Unnamed: 0'])
adult_data = adult_data.drop(columns = ['Number'])
adult_data



## Get x and y datasets- separate predictors to y outcome variables
X = adult_data[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity', 'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke', 'days_active', 'smoking_status']]
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
train_vals, test_vals = train_test_split(range(len(adult_data)), test_size = 0.2, random_state=1)
x_train = X.loc[train_vals].reset_index(drop = True)
x_test = X.loc[test_vals].reset_index(drop = True)
print(x_train[:10])
print(x_test[:10])

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
print('starting to print to csv')

x_train = OneHotEncoder().fit_transform(x_train)
x_test = OneHotEncoder().fit_transform(x_test)
## Instantiate the random forest


rsf = RandomSurvivalForest(n_estimators= 150,min_samples_split= 25,
                                           min_samples_leaf= 10,
                                           max_features= 6,
                                           n_jobs=-1,
                                           random_state= None)
estimate = rsf.fit(x_train, y_train)
pred = rsf.predict_survival_function(x_test)
cstat = rsf.score(x_test, y_test)
print('testing score: ', cstat)

### Get train score c-statistic
traincstat = rsf.score(x_train, y_train)
print('c-stat for training data: ',traincstat)


## Import data and removing variables
adult_data2 = pd.read_csv('adult_datatest2.csv')
#adult_data2 = adult_data2.drop(columns=['Unnamed: 0'])
adult_data2 = adult_data2.drop(columns = ['Number'])
adult_data2

## Get x and y datasets- separate predictors to y outcome variables
X2 = adult_data2[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity',  'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke', 'days_active', 'smoking_status']]
y2 = adult_data2[['mortstat', 'permth_int']]

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

X2 = OneHotEncoder().fit_transform(X2)


with open ('rsfcalibrate_results2.csv', 'w', newline = '') as outfile1:
    writer = csv.writer(outfile1)
    headers = ['index', 'riskscore']
    first = headers
    writer.writerow(first)
    res = []
    riskexample = pd.Series(rsf.predict(x_test))
    for i,v in riskexample.items():
        res.append(i)
        res.append(v)
        writer.writerow(res)
        res = []
    print('added all risk values to csv list')
    print(riskexample)
    #example = rsf.predict_survival_function(x_test, return_array = True)

    #writer.writerow(res)



riskexample = pd.Series(rsf.predict(X))
#print(riskexample)
example = rsf.predict_survival_function(X2, return_array = True)


for i, j in enumerate(example):
    plt.step(rsf.event_times_, j, where="post", label=str(i))
plt.ylabel("Survival probability P(T>t)")
plt.xlabel("Time (months)")
plt.title('RSF estimate survival for 6 test individuals')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('rsfsurvfunctest2.png')

###Get the survival values for calibration plot
rsfcalibrate = rsf.predict_survival_function(x_test, return_array = True)
rsfcalibrate = pd.DataFrame(data= rsfcalibrate)
rsfcalibrate.to_csv('rsfcalibrate2.csv')
print('made csv file for calibration plot')