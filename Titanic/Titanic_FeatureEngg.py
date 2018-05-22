# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:06:38 2018

@author: Sujatha Chittiri
"""

import os
import pandas as pd 
from sklearn import tree
from sklearn import model_selection
from sklearn import preprocessing


os.chdir("C:\\Users\\KK\\Python Programs\\Data")
titanic_train = pd.read_csv("titanic_train.csv")
titanic_test = pd.read_csv("titanic_test.csv")

#combine Train and Test
titanic_test['Survived'] = None
titanic = pd.concat([titanic_train,titanic_test])

#Extract Titles from the Passenger names of Train and Test Data
def ExtractTitle(name):
    return name.split(',')[1].split('.')[0].strip()
    

    
titanic['Title'] = titanic['Name'].map(ExtractTitle)

#Apply Imputer to pass default Mean values to missing Age and Fare Columns

mean_Imputer = preprocessing.Imputer()
mean_Imputer.fit(titanic_train[['Age','Fare']])
#Age is missing in both train and test data.
#Fare is NOT missing in train data but missing test data. Since we are playing on tatanic union data, we are applying mean imputer on Fare as well..
titanic[['Age','Fare']] = mean_Imputer.transform(titanic[['Age','Fare']])


#Add SibSp and Parch to get the FamilySize

titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

#Convert FamilySize to Categorical Column
def convert_FamilySize(size):
    if size == 1:
        return 'Single'
    elif size <= 3:
        return 'Small'
    elif size <= 6:
        return 'Medium'
    else:
        return 'Large'
    
titanic['FamilySize1'] = titanic['FamilySize'].map(convert_FamilySize)
        
#Convert Age to Categorical Column

def convert_Age(age):
    if age >= 0 and age <= 10:
        return 'Child'
    elif age <= 25:
        return 'Young'
    elif age <= 50:
        return 'Middle'
    else:
        return 'Old'
    
titanic['Age1'] = titanic['Age'].map(convert_Age)
    

titanic.shape
titanic.info()
titanic.describe

#Apply get_dummies
titanic1 = pd.get_dummies(titanic,columns=['Pclass', 'Sex', 'Embarked','Title','Age1','FamilySize1'])

#Drop Unwanted Columns
titanic2 = titanic1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],1)

#Take only X Train and y Train
X_titanic_train = titanic2[0:titanic_train.shape[0]]  #0 to 891 records
y_titanic_train = titanic_train['Survived']

#DecisionTree
dt = tree.DecisionTreeClassifier(random_state=2017)
param_grid = {'max_depth':list(range(1,6)),'min_samples_split':[2,3,5,7],'criterion':['gini','entropy']}
grid_tree_estimator = model_selection.GridSearchCV(dt,param_grid, cv=10, n_jobs=2)

grid_tree_estimator.fit(X_titanic_train,y_titanic_train)

print(grid_tree_estimator.grid_scores_)
print(grid_tree_estimator.best_score_)
print(grid_tree_estimator.best_params_)
print(grid_tree_estimator.score(X_titanic_train,y_titanic_train))

features = X_titanic_train.columns
importances = grid_tree_estimator.best_estimator_.feature_importances_
fe_df = pd.DataFrame({'feature':features,'importance':importances})
#Now let's predict on test data
X_titanic_test = titanic2[titanic_train.shape[0]:] #shape[0]: means 0 index to n index. Not specifying end index is nothing but till nth index
X_titanic_test.shape
X_titanic_test.info()
titanic_test['Survived'] = grid_tree_estimator.predict(X_titanic_test)

titanic_test.to_csv('submission_FeatureEngg4.csv', columns=['PassengerId','Survived'],index=False)