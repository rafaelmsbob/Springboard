# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:47:24 2020

@author: Rafael
"""
#This code will use the best parameters calculated by GridSearchCV to generate a scatter plot of the predictions X actual values.
#As we do not have the actual values of the testing dataset of the competition, this code will perform a train/test split,
#reserving 20% for testing, which will not be used for fitting the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

#read training dataset
df = pd.read_csv('train.csv', header=0, index_col='Id')

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

#Move target column to the end
aux = df.SalePrice
df.drop(labels=['SalePrice'], axis=1, inplace = True)
df.insert(len(df.columns), 'SalePrice', aux)

#split data
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=0.2, random_state=0)

#steps for the pipeline
steps = [('scaler', StandardScaler()), ('gradient', GradientBoostingRegressor(random_state=0, n_estimators=1000, learning_rate=0.05))]

#Assemble pipeline
pipeline = Pipeline(steps)

#fit the model
pipeline.fit(X_train, y_train)

#scatter plot----------------
#prediction
y_pred = pipeline.predict(X_test)

#scatter
plt.scatter(y_test/1000, y_pred/1000)

#iso line
plt.plot([0,800],[0,800], '--', color='red')

#decorations
plt.xlabel('Actual Values (/1,000)')
plt.xticks(np.arange(0,800,100))
plt.ylabel('Predicted Values (/1,000)')
plt.yticks(np.arange(0,800,100))
plt.title('Predicted X Actual Values')

plt.show()





























