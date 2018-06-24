# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:51:09 2018

@author: KK
"""

import os
import pandas as pd
from sklearn import model_selection
from sklearn import metrics 
import math
from xgboost import XGBRegressor


os.chdir("C:\\Users\\KK\\Python Programs\\Data")
rest_train = pd.read_csv("rest_train.csv")
rest_test = pd.read_csv("rest_test.csv")

rest_train.shape
rest_train.describe()
rest_train.info()

rest_test.info()
rest_data = pd.concat([rest_train,rest_test],ignore_index=True)

#rest_train1  = pd.get_dummies(rest_data,columns=["City","City Group","Type","P29"])

#Split Year, Month and Date from Open Data

def SplitDate(date):
    return date.split('/')[1].split('/')[0].strip()

def SplitYear(date):
    return date.split('/')[2].strip()

def SplitMonth(date):
    return date.split('/')[0].strip()

rest_data['Date'] = rest_data['Open Date'].map(SplitDate)
rest_data['Year'] = rest_data['Open Date'].map(SplitYear)
rest_data['Month'] = rest_data['Open Date'].map(SplitMonth)

#Categorise years into Old, Middle and recent
def GroupYear(year):
    if  year < '2000':
        return 'Old'
    elif year < '2010':
        return 'Middle'
    else:
        return 'Recent'
   
rest_data['GroupYear'] = rest_data['Year'].map(GroupYear)

rest_data['IsIstanbul'] = rest_data["City"] == 'İstanbul'
rest_train1  = pd.get_dummies(rest_data,columns=["City","City Group","Type","GroupYear"])

rest_train2 = rest_train1.drop(["revenue","Open Date","Date","Year","Month"],1)

X_train = rest_train2[0:rest_train.shape[0]]

y_train = rest_train["revenue"]

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )

xgb_regressor = XGBRegressor(seed=10)

xgb_grid = {'learning_rate': [0.2, 0.3, 0.5, 0.7],
               'n_estimators': range(150, 201, 50),
               'max_depth': range(15, 21, 5),
               'reg_alpha': [0.55, 0.6, 0.65, 0.7],
               'reg_lambda': [0.45, 0.5, 0.55, 0.6]}

xgb_grid_estimator = model_selection.GridSearchCV(xgb_regressor, xgb_grid, scoring = metrics.make_scorer(rmse), cv=10, n_jobs=1)

xgb_grid_estimator.fit(X_train,y_train)

xgb_grid_estimator.score(X_train,y_train)
xgb_grid_estimator.best_score_
xgb_grid_estimator.best_params_

X_test = rest_train2[rest_train.shape[0]:]
X_test.shape

rest_test['Prediction'] = xgb_grid_estimator.predict(X_test)
rest_test.to_csv("RestaurantXGB2.csv",columns=["Id","Prediction"],index=False)

