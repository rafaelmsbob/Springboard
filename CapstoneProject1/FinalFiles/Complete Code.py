# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:47:24 2020

@author: Rafael
"""
#this is the code that brings the training and testing dataset all the way from raw csv files to the result of
#a GradientBoosting predictions, generating a file to submit to the Kaggle competition.

#several methods were tested before coming to Gradient Boosting. Those attempts are not in this code, please refer
#to the codes snippets presented in the FInal report, which can be found in the same folder on my github rep

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

#code logic:
#1) Separate last column (target column - SellingPrice) from the training set. (testing set don't have that column) 
#2) Append testing rows to training set
#3) Data Wrangling
#4) Separate train and test
#5) run ML model

#read training dataset
df = pd.read_csv('train.csv', header=0, index_col='Id')

#separate y_train (target column)
y_train = df.iloc[:, -1]
df = df.iloc[:, 0:-1]

#read testing dataset
df_test = pd.read_csv('test.csv', header=0, index_col='Id')

#save the number of the last row in the training dataset. Useful to unappend later
lastRow = df.shape[0]

#append test dataset
df = df.append(df_test)

#------Data Wrangling------
#replace typos
df.BldgType = df.BldgType.replace('Twnhs', 'TwnhsI')
df.BldgType = df.BldgType.replace('Duplex', 'Duplx')
df.BldgType = df.BldgType.replace('2fmCon', '2FmCon')

df.MSZoning = df.MSZoning.replace('C (all)', 'C')

df.Exterior2nd = df.Exterior2nd.replace('Wd Shng', 'WdShing')
df.Exterior2nd = df.Exterior2nd.replace('CmentBd', 'CemntBd')
df.Exterior2nd = df.Exterior2nd.replace('Brk Cmn', 'BrkComm')

df.Neighborhood = df.Neighborhood.replace('NAmes', 'Names')

df.MasVnrArea = df.MasVnrArea.replace(1.0,0.0)

for index, col in df.iterrows():
    if((col['MasVnrType'] == 'None') & (col['MasVnrArea'] > 0)):
        df.loc[index, 'MasVnrType'] = 'BrkFace'
df = df.replace({'MasVnrType':{np.nan : 'None'}, 'MasVnrArea': {np.nan : 0}}, value = None)

#identifying columns with null values
nulls = df.isnull().sum()

#read file with all categorical columns names.
dfAllCatColumns = pd.read_csv('AllPossibleValuesForCatCols.csv', header=0).columns

#using set operations to identify the categorical columns that have null values
catcolswithnan = set(dfAllCatColumns) & set(nulls[nulls>0].index)       

#replace missing values with "No Item" for the cateforical columns
for col in catcolswithnan:
    df[col] = df[col].replace(np.nan, 'No Item')    

#assuming every house has electricity. Missing values are replaced by the standard electrical system
df.Electrical = df.Electrical.replace('No Item', 'SBrkr')

#fill in the missing values for LotFrontage area with a LinearRegression model
#save first and last index
firstIdx = df.index[0]
lastIdx = df.index[-1]

#filter out NaN
dfLotTrain = df[np.logical_not(np.isnan(df['LotFrontage']))]
dfLotTest = df[np.isnan(df['LotFrontage'])]

#apply method
model = LinearRegression()
model.fit(dfLotTrain['LotArea'].values.reshape(-1, 1), dfLotTrain['LotFrontage'].values.reshape(-1, 1))
aux = model.predict(dfLotTest['LotArea'].values.reshape(-1, 1))
dfLotTest = dfLotTest.assign(LotFrontage=aux)

#append rows from dfTest to regenerate df
df = dfLotTrain.append(dfLotTest)
#df = df.reindex(list(range(1,df.shape[0]+1)))
df = df.reindex(list(range(firstIdx, lastIdx+1)))

#GarageYrBlt is a column that shows NaN when the property has no Garage. For those houses, I decided to penalize 
#them by increasing the age of the "garage" by 10 years

for index, col in df.iterrows():    
    if(np.isnan(col['GarageYrBlt'])):
        df.loc[index, 'GarageYrBlt'] = df.loc[index, 'YearBuilt'] - 10

#create column with year intervals
df['YearsSinceLastRemod'] = df.YrSold - df['YearRemodAdd']
df['YearsSinceBuilt'] = df.YrSold - df['YearBuilt']
df['YearsSinceGarageBuilt'] = df.YrSold - df['GarageYrBlt']

#if there's still NaN, they are in numerical columns. Fill with 0
df = df.fillna(0)

#generate dummy columns
df = pd.get_dummies(df, drop_first = True)

#split back to train/test
X_train = df.iloc[:lastRow,:]
X_test = df.iloc[lastRow:,:]

#steps for the pipeline
#steps = [('scaler', StandardScaler()), ('gradient', GradientBoostingRegressor(random_state=0))]
steps = [('scaler', StandardScaler()), ('gradient', GradientBoostingRegressor(random_state=0))]
pipeline = Pipeline(steps)

#parameters for the gradient boosting regressor
parameters = {'gradient__learning_rate':[0.01, 0.05, 0.1], 'gradient__n_estimators':[500, 1000, 1500]}


reg = GridSearchCV(pipeline, param_grid=parameters, cv=5, scoring='r2')
reg.fit(X_train, y_train)

#generate csv file for submission
result = pd.DataFrame({'Id':X_test.index, 'SalePrice': reg.predict(X_test)})
result.to_csv("submission_Rafael_gradBoostCV.csv", index=False)