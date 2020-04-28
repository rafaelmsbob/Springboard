# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:57:48 2020

@author: Rafael
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv', header=0, index_col='Id')
dfAll = pd.read_csv('AllPossibleValuesForCatCols.csv', header=0)
dfAll = dfAll.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
#dfnumbers = df._get_numeric_data()
#catcols = set(df.columns) - set(dfnumbers.columns)
#catcols = catcols.union(set(['MSSubClass','OverallQual','OverallCond']))
    
#print(nulls[nulls>0].sort_values(ascending=False))    #no null value 
#print(nulls[nulls>0].index)
#dfnumbers = df._get_numeric_data()
#print(df._get_numeric_data())
#print(dfnumbers.columns)

#nans = df[df.isna()].count()
#print(nans[nans>0])             
#print(nans)

df.BldgType = df.BldgType.replace('Twnhs', 'TwnhsI')
df.BldgType = df.BldgType.replace('Duplex', 'Duplx')
df.BldgType = df.BldgType.replace('2fmCon', '2FmCon')

df.MSZoning = df.MSZoning.replace('C (all)', 'C')

df.Exterior2nd = df.Exterior2nd.replace('Wd Shng', 'WdShing')
df.Exterior2nd = df.Exterior2nd.replace('CmentBd', 'CemntBd')
df.Exterior2nd = df.Exterior2nd.replace('Brk Cmn', 'BrkComm')

df.Neighborhood = df.Neighborhood.replace('NAmes', 'Names')

#diffs = {}
#
#for col in catcols:
#    diffs[col] = set(df[col]) - set(dfAll[col])
#    
#for key, value in diffs.items():
#    if(len(value)>0): print(key, ' - ',value)

df.MasVnrArea = df.MasVnrArea.replace(1.0,0.0)
#df.MasVnrType = df[(df.MasVnrType == 'None') & (df.MasVnrArea > 0)].MasVnrType.replace('None', 'brkFace')

for index, col in df.iterrows():
    if((col['MasVnrType'] == 'None') & (col['MasVnrArea'] > 0)):
        df.loc[index, 'MasVnrType'] = 'BrkFace'
df = df.replace({'MasVnrType':{np.nan : 'None'}, 'MasVnrArea': {np.nan : 0}}, value = None)

nulls = df.isnull().sum()
catcolswithnan = set(dfAll.columns) & set(nulls[nulls>0].index)    
   
#print(df[['MasVnrType', 'MasVnrArea']][(df.MasVnrType == 'None') & (df.MasVnrArea > 0)])    
#print(df[['MasVnrType', 'MasVnrArea']][df.MasVnrArea > 0].MasVnrType.value_counts())

for col in catcolswithnan:
    df[col] = df[col].replace(np.nan, 'No Item')
    #print(col, ' - ',df[col].unique())
    
#print(df[['Electrical', 'SalePrice']].groupby('Electrical').mean())

#df[['Electrical', 'SalePrice']].groupby('Electrical').mean().plot(kind = 'bar')
#plt.show()

df.Electrical = df.Electrical.replace('No Item', 'SBrkr')

#for col in catcolswithnan:
#    print(col, ' - ',df[col].unique())

dfNumericCols = set(df.columns) - set(dfAll.columns)
dfNumeric = df[list(dfNumericCols)]
#print(dfNumericCols)
#print(dfNumeric.isnull().sum()[dfNumeric.isnull().sum()>0])

#print(df[['GarageType', 'GarageYrBlt']][df.GarageYrBlt.isnull()].GarageType.unique())
#print(dfNumeric[dfNumeric.isnull().sum()>0])

#df[['LotFrontage', 'LotArea']].plot(x = 'LotFrontage', y = 'LotArea',xlim=(0,150), ylim=(0,20000), kind = 'scatter')
#plt.show()

#print(df[['LotFrontage', 'LotArea']].corr())

aux = 0

for index, col in df.iterrows():    
    if(np.isnan(col['LotFrontage'])):        
        aux = 0.0046*df.loc[index, 'LotArea'] + 27.113 
        if(aux>200):
            df.loc[index, 'LotFrontage'] = 0.0006*df.loc[index, 'LotArea'] + 19.657
        else:
            df.loc[index, 'LotFrontage'] = aux
            
df= df[df.LotFrontage < 200] 
#df= df[df.LotArea < 50000]

#print(df.MasVnrArea.isnull().sum())
#        
#df[['LotFrontage', 'LotArea']].plot(x = 'LotFrontage', y = 'LotArea', kind = 'scatter')
#plt.show()

#print(df[['MiscFeature', 'MiscVal']][(df.MiscFeature == 'No Item') & (df.MiscVal != 0 )])
#print(df[['PoolArea', 'PoolQC']][(df.PoolQC == 'No Item') & (df.PoolArea != 0 )])
#print(df[['Fireplaces', 'FireplaceQu']][(df.FireplaceQu == 'No Item') & (df.Fireplaces != 0 )])

#collist = ['GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']
#
#df.GarageType== 'No Item'
#myDict = {}
#
#for col in collist:
#    myDict[col] = []
#
#for index, row in df.iterrows():
#    if(df.loc[index, 'GarageType'] == 'No Item'):
#        for col in collist:            
#            myDict[col].append(row[col])
#
#dfGarage = pd.DataFrame(myDict)
#
#for col in dfGarage.columns:
#    print(col, '- ', dfGarage[col].unique())

#collist = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
#
#df.BsmtQual == 'No Item'
#myDict = {}
#
#for col in collist:
#    myDict[col] = []
#
#for index, row in df.iterrows():
#    if(df.loc[index, 'BsmtQual'] == 'No Item'):
#        for col in collist:            
#            myDict[col].append(row[col])
#
#dfBsmt = pd.DataFrame(myDict)
#
#for col in dfBsmt.columns:
#    print(col, '- ', dfBsmt[col].unique())
#for col in dfNumericCols:
#    ax = sns.violinplot(y=col, data=df)
#    plt.show()

ax = sns.violinplot(y='MasVnrArea', data=df[df.MasVnrArea > 0])