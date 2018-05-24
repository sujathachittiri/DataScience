# -*- coding: utf-8 -*-
"""
Created on Tue May  1 07:51:29 2018

@author: Sujatha Chittiri
"""
import os
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import metrics 
import math

os.chdir("C:\\Users\\KK\\Python Programs\\Data")
rest_train = pd.read_csv("rest_train.csv")
rest_test = pd.read_csv("rest_test.csv")

rest_train.shape
rest_train.describe()
rest_train.info()

rest_test.info()
rest_data = pd.concat([rest_train,rest_test],ignore_index=True)

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

rest_train1  = pd.get_dummies(rest_data,columns=["City","City Group","Type","P29","GroupYear"])

rest_train2 = rest_train1.drop(["revenue","Open Date"],1)

X_train = rest_train2[0:rest_train.shape[0]]

y_train = rest_train["revenue"]

#RandomForestRegressor

rf=ensemble.RandomForestRegressor(random_state = 2014)
parm_grid = {'n_estimators': [5,15,25], 'max_depth':[3,7,10]}

rf_grid_estimator = model_selection.GridSearchCV(rf,parm_grid,cv=10)

rf_grid_estimator.fit(X_train,y_train)
rf_grid_estimator.score(X_train,y_train)
rf_grid_estimator.best_score_
rf_grid_estimator.best_params_

X_train.shape
 
#Feature Importances

features = rest_train2.columns
importances = rf_grid_estimator.best_estimator_.feature_importances_
fe_df = pd.DataFrame({'feature':features,'importance':importances})
fe_df.sort_values(by='importance', ascending = False)
fe_df.set_index('feature', drop=False)
fe_df.plot.barh(title='Feature Importances', figsize=(200,200))

fs_model = feature_selection.SelectFromModel(rf_grid_estimator.best_estimator_, threshold="mean", prefit=True)
X_train1 = fs_model.transform(rest_train2)

X_train1.shape
rest_train2.shape

X_train2 = X_train1[0:rest_train.shape[0]]

#Apply KNN on the selected features 

scaled_model = preprocessing.StandardScaler()
scaled_model.fit(X_train2)
X_train2 = scaled_model.transform(X_train2)
knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':[15, 20]}

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )

knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, scoring = metrics.make_scorer(rmse), cv=10, n_jobs=1)

knn_grid_estimator.fit(X_train2,y_train)

knn_grid_estimator.score(X_train2,y_train)
knn_grid_estimator.best_score_
knn_grid_estimator.best_params_

X_test = X_train1[rest_train.shape[0]:]
X_test.shape

rest_test['Prediction'] = knn_grid_estimator.predict(X_test)
rest_test.to_csv("RestaurantRFKNN.csv",columns=["Id","Prediction"],index=False)

