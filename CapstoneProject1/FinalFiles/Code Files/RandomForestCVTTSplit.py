# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:47:24 2020

@author: Rafael
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

#read training file
df = pd.read_csv('DatasetTreated.csv', index_col=0, header=0)

#Separate last column (target)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

#generate dummy columns
X = pd.get_dummies(X, drop_first = True)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#run the model
steps = [('scaler', StandardScaler()), ('RF', RandomForestClassifier(random_state=0))]
pipeline = Pipeline(steps)

parameters = {'RF__max_depth':[7,8], 'RF__n_estimators':[200,100]}
#parameters = {'l1_ratio':np.linspace(0,1,100)}

reg = GridSearchCV(pipeline, param_grid=parameters, cv=5)
reg.fit(X_train, y_train)

#calculate Score
print(reg.score(X_test, y_test))
print(reg.best_params_)