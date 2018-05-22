# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:35:49 2018

@author: Sujatha Chittiri
"""

import os
import pandas as pd 
from sklearn import tree
from sklearn import decomposition

os.getcwd()
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.getcwd()
os.chdir("C:\\Users\\KK\\Python Programs\\Data")
titanic_train = pd.read_csv("titanic_train.csv")

#EDA

titanic_train.shape
titanic_train.info()

#Transformation of non numneric cloumns
#There is an exception with the pclass. Though it's coinncidentally is a number but it's a classification but not a number.
#titanic_train1 = titanic_train[['Pclass', 'Sex', 'Embarked', 'Fare']]

#Convert categoric to One hot encoding using get_dummies

titanic_train1 = pd.get_dummies(titanic_train,columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe

#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_titanic_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],1)
y_titanic_train = titanic_train1['Survived']

pca = decomposition.PCA(n_components=4)
pca.fit(X_titanic_train)

X_transformed_Train = pca.transform(X_titanic_train)

dt = tree.DecisionTreeClassifier()
dt.fit(X_transformed_Train,y_titanic_train)

titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass', 'Sex', 'Embarked'])
X_titanic_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'],1)

pca = decomposition.PCA(n_components=4)
pca.fit(X_titanic_test)

X_transformed_Test = pca.transform(X_titanic_test)

#Apply the model on Furture/test data

titanic_test['Survived'] = dt.predict(X_transformed_Test)
titanic_test.to_csv("Submission_PCA.csv",columns=['PassengerId','Survived'],index=False)
