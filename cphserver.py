#import pysurvival
import csv
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.utils.metrics import concordance_index

## add absolute path file
adult_data = pd.read_csv('adult_data.csv')
#adult_data

adult_data = adult_data.drop(columns=['Unnamed: 0'])
adult_data = adult_data.drop(columns = ['Number'])
print('got data and removed unnecessary columns...so far so good...')

## Create test and train sets
train_vals, test_vals = train_test_split(range(len(adult_data)), test_size = 0.2)
train_data = adult_data.loc[train_vals].reset_index(drop = True)
test_data = adult_data.loc[test_vals].reset_index(drop = True)

## Collect predictor variables from dataset
predictors = list(adult_data[['BMI', 'Systolic', 'Diastolic', 'regularity', 'Chol2', 'Ethnicity', 'Household_income', 'Gender', 'Age', 'heart_attack', 'relative_ha', 'liver_problem', 'cancer', 'stroke', 'kidney_disease', 'occupation', 'days_active', 'smoking_status']])

## Set-up test and train datasets using predictors and time and event columns
X_train, X_test = train_data[predictors], test_data[predictors]
Time_train, Time_test = train_data['permth_int'].values, test_data['permth_int'].values
Event_train, Event_test = train_data['mortstat'].values, test_data['mortstat'].values

from itertools import combinations
from more_itertools import powerset, ilen
#time = adult_data['permth_int']
#event = adult_data['mortstat']


combos2 = powerset(predictors)
comboslist = list(combos2)
len(comboslist)

## remove model combos with 4 or fewer variables
removes = []
for x in comboslist:
    if len(x) <= 4:
        val = comboslist.index(x)
        removes.append(val)
removes[-2]

del comboslist[0:4048]

print(len(comboslist))

## loop through the variables and make best model and return best concordance value and predictors for the best model
## need to add this to a csv to get it to return the details- use RSF as basis for this code

## Different instance of Cox PH
coxph = CoxPHFitter()

with open ('coxph_results.csv', 'w', newline = '') as outfile1:
    writer = csv.writer(outfile1)
    headers = ['cstatistic', 'AIC', 'model_preds']
    first = headers
    writer.writerow(first)

    model_vals = []
    current = 0
    for combo in comboslist:
	combo = list(combo)
	combo.append('permth_int')
	combo.append('mortstat')
	print(combo)
        preds = train_data[combo] 
        print(preds)
        model = coxph.fit(preds, duration_col= 'permth_int', event_col='mortstat')
        cstat = coxph.concordance_index_
        AIC = coxph.AIC_partial_
        if cstat > current:
            current = cstat
            var = preds
            model_vals.append(current)
            model_vals.append(AIC)
            for v in var:
                model_vals.append(v)
            writer.writerow(model_vals)
        else:
            continue