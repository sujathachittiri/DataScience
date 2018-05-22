# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 08:41:29 2018

@author: Sujatha Chittiri
"""

import os
import pandas as pd 
from sklearn import tree
import io
import pydot
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


#build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train,y_titanic_train)

#visualize the decission tree
dot_data = io.StringIO()
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_titanic_train.columns)
graph=pydot.graph_from_dot_data
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_pdf("DS-Augmented.pdf")

#Predict the outcome using decision tree
titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test,columns=['Pclass', 'Sex', 'Embarked'])
X_titanic_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'],1)

#Apply the model on Furture/test data

titanic_test['Survived'] = dt.predict(X_titanic_test)
titanic_test.to_csv("Submission23MarV2.csv",columns=['PassengerId','Survived'],index=False)
