# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:09:45 2018

@author: Sujatha Chittiri
"""
import os
import pandas as pd 
#from sklearn import tree
#from sklearn import model_selection
#from sklearn.model_selection import GridSearchCV
#import io
#import pydot
from sklearn.externals import joblib

os.chdir("C:\\Users\\KK\\Python Programs\\Data")

dtree=joblib.load("tree1.pkl")
#Predict the outcome using decision tree
titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass', 'Sex', 'Embarked'])
X_titanic_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'],1)

#Apply the model on Furture/test data

titanic_test['Survived'] = dtree.predict(X_titanic_test)
titanic_test.to_csv("SubmissionwithJoblib.csv",columns=['PassengerId','Survived'],index=False)
