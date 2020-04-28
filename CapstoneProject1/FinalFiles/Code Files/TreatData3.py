# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:57:48 2020

@author: Rafael
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#read train or test dataset
df = pd.read_csv('test.csv', header=0, index_col='Id')

#read file with all categorical columns names.
dfAllCatColumns = pd.read_csv('AllPossibleValuesForCatCols.csv', header=0).columns

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
#print(df.LotFrontage.isnull().sum())
nulls = df.isnull().sum()


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
dfTrain = df[np.logical_not(np.isnan(df['LotFrontage']))]
dfTest = df[np.isnan(df['LotFrontage'])]

#apply method
model = LinearRegression()
model.fit(dfTrain['LotArea'].values.reshape(-1, 1), dfTrain['LotFrontage'].values.reshape(-1, 1))
aux = model.predict(dfTest['LotArea'].values.reshape(-1, 1))
dfTest = dfTest.assign(LotFrontage=aux)

#append rows from dfTest to regenerate df
df = dfTrain.append(dfTest)
#df = df.reindex(list(range(1,df.shape[0]+1)))
df = df.reindex(list(range(firstIdx, lastIdx+1)))

#GarageYrBlt is a column that shows NaN when the property has no Garage. For those houses, I decided to penalize 
#them by increasing the number of years the garage was built by 10 years

for index, col in df.iterrows():    
    if(np.isnan(col['GarageYrBlt'])):
        df.loc[index, 'GarageYrBlt'] = df.loc[index, 'YearBuilt'] - 10

#create column with year intervals
df['YearsSinceLastRemod'] = df.YrSold - df['YearRemodAdd']
df['YearsSinceBuilt'] = df.YrSold - df['YearBuilt']
df['YearsSinceGarageBuilt'] = df.YrSold - df['GarageYrBlt']

#move SalePrice column to the end (for the training dataset that has the target column SalePrice)
#mid = df['SalePrice']
#df.drop(labels=['SalePrice'], axis=1, inplace = True)
#df['SalePrice'] = mid

#if there's still NaN, they are in numerical columns. Fill with 0
df = df.fillna(0)

#Generate csv with dataset treated
df.to_csv("DatasetTreated_test.csv", index=True)