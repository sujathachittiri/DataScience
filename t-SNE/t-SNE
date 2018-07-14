import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE #t-Distributed Stochastic Neighbor Embedding

#changes working directory
os.chdir("D:/Data Science/Data")

titanic_train = pd.read_csv("titanic_train.csv") #11d Data

#EDA
titanic_train.shape
titanic_train.info()

#User defined function to plot the data
def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1)
    plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
    plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
    plt.legend()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
X_train.shape

#tSNE-->t-Distributed Stochastic Neighbor Embedding
#tSNE plotting helps to understand whethe the data in liner or non-liner
#For perplexity: Refer to http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html
tsne = TSNE(perplexity=30.0, n_components=2, n_iter=250)
X_train.info()
titanic_2 = tsne.fit_transform(X_train)

plot_data(titanic_2, np.array(titanic_train1['Survived']))
