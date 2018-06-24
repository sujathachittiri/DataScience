# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:21:27 2018

@author: KK
"""

import os
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection

os.chdir("C:\\Users\\KK\\Python Programs\\Data")
rest_train = pd.read_csv("rest_train.csv")
rest_test = pd.read_csv("rest_test.csv")

rest_train.shape
rest_train.describe()
rest_train.info()

rest_test.info()
rest_data = pd.concat([rest_train,rest_test],ignore_index=True)

rest_train1  = pd.get_dummies(rest_data,columns=["City","City Group","Type"])

#Split Year, Month and Date from Open Data

def SplitDate(date):
    return date.split('/')[1].split('/')[0].strip()

def SplitYear(date):
    return date.split('/')[2].strip()

def SplitMonth(date):
    return date.split('/')[0].strip()

rest_train1['Date'] = rest_train1['Open Date'].map(SplitDate)
rest_train1['Year'] = rest_train1['Open Date'].map(SplitYear)
rest_train1['Month'] = rest_train1['Open Date'].map(SplitMonth)


rest_train2 = rest_train1.drop(["revenue","Open Date"],1)

X_train = rest_train2[0:rest_train.shape[0]]

y_train = rest_train["revenue"]

rf=ensemble.RandomForestRegressor(random_state = 2014)
parm_grid = {'n_estimators': [5,10], 'max_depth':[6,8]}

rf_grid_estimator = model_selection.GridSearchCV(rf,parm_grid,cv=10)

rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.score(X_train,y_train)
rf_grid_estimator.best_score_
rf_grid_estimator.best_params_

X_test = rest_train2[rest_train.shape[0]:]

rest_test['Prediction'] = rf_grid_estimator.predict(X_test)

rest_test.to_csv("RestaurantRF.csv",columns=["Id","Prediction"],index=False)


