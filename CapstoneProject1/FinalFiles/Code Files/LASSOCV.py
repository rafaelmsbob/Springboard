# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:47:24 2020

@author: Rafael
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

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

#reg = LassoCV(cv=5, alphas=[10, 60, 100, 200, 500, 1000], max_iter=10000, random_state=0).fit(X_train, y_train)
reg = LassoCV(cv=5, alphas=[110], max_iter=10000, random_state=0).fit(X_train, y_train)


df_feat = pd.DataFrame(index=X_aux.columns, data={'Reg_Coef':reg.coef_})
df_feat['Signal'] = df_feat.apply(lambda x: 'Pos' if(x['Reg_Coef']>0) else 'Neg', axis=1)
df_feat['Abs'] = df_feat.apply(lambda x: np.abs(x['Reg_Coef']), axis=1)
df_feat = df_feat.sort_values(by=['Abs'], ascending=False).head(15)

plt.bar(df_feat.index, df_feat.Abs, color=df_feat.Signal.map({'Pos':'blue', 'Neg':'red'}))
plt.xticks(rotation=60)
plt.figure(figsize=(20,40))
plt.show()

#print(X_train.columns)
#print(X_test.columns)
#print(reg.predict(X_test))

result = pd.DataFrame({'Id':X_test.index, 'SalePrice': reg.predict(X_test)})
result.to_csv("submission_Rafael_LassoCVBest.csv", index=False)