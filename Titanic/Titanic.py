# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:00:58 2018

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
titanic_train = pd.read_csv("C:\\Users\\KK\\Python Programs\\titanic_train.csv")

#EDA

titanic_train.shape
titanic_train.info()

X_titanic_train = titanic_train[['Pclass','Parch']]
y_titanic_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train,y_titanic_train)

#visualize the decission tree
dot_data = io.StringIO()
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_titanic_train.columns)
graph=pydot.graph_from_dot_data
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_pdf("DS-DT.pdf")

#Predict the outcome using decision tree
titanic_test = pd.read_csv("C:\\Users\\KK\\Python Programs\\titanic_test.csv")
X_titanic_test = titanic_test[['Pclass','Parch']]
titanic_test['Survived'] = dt.predict(X_titanic_test)
titanic_test.to_csv("Submission23MarV1.csv",columns=['PassengerId','Survived'],index=False)
