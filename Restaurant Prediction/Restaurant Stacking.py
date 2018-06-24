# -*- coding: utf-8 -*-
"""
Created on Wed May  2 19:46:46 2018

@author: KK
"""

import os
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import neighbors
from mlxtend import regressor

os.chdir("C:\\Users\\KK\\Python Programs\\Data")
rest_train = pd.read_csv("rest_train.csv")
rest_test = pd.read_csv("rest_test.csv")

rest_train.shape
rest_train.describe()
rest_train.info()

rest_test.info()
rest_data = pd.concat([rest_train,rest_test],ignore_index=True)

#rest_train1  = pd.get_dummies(rest_data,columns=["City","City Group","Type"])

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
rest_train1  = pd.get_dummies(rest_data,columns=["City","City Group","Type","P29","GroupYear"])

rest_train2 = rest_train1.drop(["revenue","Open Date"],1)

X_train = rest_train2[0:rest_train.shape[0]]

y_train = rest_train["revenue"]

rf=ensemble.RandomForestRegressor(random_state = 2014)
parm_grid = {'n_estimators': [5,15,25], 'max_depth':[3,7,10]}

rf_grid_estimator = model_selection.GridSearchCV(rf,parm_grid,cv=10)

rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.score(X_train,y_train)
rf_grid_estimator.best_score_
rf_grid_estimator.best_params_

#features = X_train.columns
features = rest_train2.columns
importances = rf_grid_estimator.best_estimator_.feature_importances_
fe_df = pd.DataFrame({'feature':features,'importance':importances})
fe_df.sort_values(by='importance', ascending = True, inplace=True)
fe_df.set_index('feature', inplace=False)
fs_model = feature_selection.SelectFromModel(rf_grid_estimator.best_estimator_, prefit=True)

X_train1 = fs_model.transform(rest_train2)

X_train1.shape 
selected_features = X_train.columns[fs_model.get_support()]

X_train2 = X_train1[0:rest_train.shape[0]]

#build stacked model using selected features
rf1 = ensemble.RandomForestRegressor(random_state=100)
knn2 = neighbors.KNeighborsRegressor()
gb3 = ensemble.GradientBoostingRegressor(random_state=100)

lr = linear_model.LogisticRegression(random_state=100)

stack_estimator = regressor.StackingRegressor(regressors=[rf1, knn2, gb3], meta_regressor=lr) #, store_train_meta_features=True)
stack_grid = {'randomforestregressor__n_estimators': [3,6,9,14],
            'kneighborsregressor__n_neighbors': [5, 10],
            'gradientboostingregressor__n_estimators': [10, 50],
            'meta-logisticregression__C': [0.1, 10.0]}

grid_stack_estimator = model_selection.GridSearchCV(stack_estimator, stack_grid, cv=10)
grid_stack_estimator.fit(X_train2, y_train)


X_test = X_train1[rest_train.shape[0]:]
X_test.shape

rest_test['Prediction'] = grid_stack_estimator.predict(X_test)
rest_test.to_csv("RestaurantStacking.csv",columns=["Id","Prediction"],index=False)
