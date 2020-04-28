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

df_train = pd.read_csv('DatasetTreated.csv', index_col=0, header=0) 
df_test = pd.read_csv('DatasetTreated_test.csv', index_col=0, header=0)

X_aux = df_train.iloc[:, 0:-1]
y_train = df_train.iloc[:, -1]

#save the number of the last row in the training dataset. Useful to unappend later
lastRow = X_aux.shape[0]

#append test dataset, so when the dummy columns are generated, any categorical data from the testing dataset will alse be present
X_aux = X_aux.append(df_test)

#generate dummy columns
X_aux = pd.get_dummies(X_aux, drop_first = True)
#X_aux.to_csv("cols_train_all.csv", index=False)

#split back to train/test
X_train = X_aux.iloc[:lastRow,:]
X_test = X_aux.iloc[lastRow:,:]

#generate csvs to check
#X_train.to_csv("cols_train.csv", index=False)
#X_test.to_csv("cols_test.csv", index=False)

steps = [('scaler', StandardScaler()), ('RF', RandomForestClassifier(random_state=0))]
pipeline = Pipeline(steps)

parameters = {'RF__max_depth':np.arange(2,10), 'RF__n_estimators':np.arange(100,600,100)}
#parameters = {'l1_ratio':np.linspace(0,1,100)}

reg = GridSearchCV(pipeline, param_grid=parameters, cv=5)
#reg = GridSearchCV(ElasticNet(max_iter=100000), param_grid=parameters)
reg.fit(X_train, y_train)

print(reg.best_params_)

#print(X_train.columns)
#print(X_test.columns)
#print(reg.predict(X_test))

result = pd.DataFrame({'Id':X_test.index, 'SalePrice': reg.predict(X_test)})
result.to_csv("submission_Rafael8diff.csv", index=False)