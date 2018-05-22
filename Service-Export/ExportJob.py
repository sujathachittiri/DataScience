# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:06:58 2018

@author: Sujatha Chittiri
"""

import os
import pandas as pd 
from sklearn import tree
from sklearn import model_selection
#from sklearn.model_selection import GridSearchCV
#import io
#import pydot
from sklearn.externals import joblib
os.chdir("C:\\Users\\KK\\Python Programs\\Data")

os.getcwd()
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.getcwd()
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
dt = tree.DecisionTreeClassifier(random_state=2014)

param_grid = {'max_depth':list(range(3,8)),'min_samples_split':[2,3,6,7,8],'criterion':['gini','entropy']}
grid_tree_estimator = model_selection.GridSearchCV(dt,param_grid, cv=10, n_jobs=5)

grid_tree_estimator.fit(X_titanic_train,y_titanic_train)

print(grid_tree_estimator.grid_scores_)
print(grid_tree_estimator.best_score_)
print(grid_tree_estimator.best_params_)
print(grid_tree_estimator.score(X_titanic_train,y_titanic_train))

joblib.dump(grid_tree_estimator,"Tree1.pkl")



#visualize the decission tree
#dot_data = io.StringIO()
#tree.export_graphviz(dt, out_file = dot_data, feature_names = X_titanic_train.columns)
#graph=pydot.graph_from_dot_data
#graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
#graph.write_pdf("DS-Augmented.pdf")

