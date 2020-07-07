# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:14:02 2020

@author: Rafael
"""
# Import all libraries needed
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# import Python file with the functions used in this code
import FRTPFunctions as func

# Load raw dataset
df = pd.read_csv("train.csv", header=0, index_col=0)

# Remove rows with null values
df.dropna(inplace=True)

# Convert Columns DATE_OF_BIRTH and DISBURSAL_DATE from string to DateTime format
df['DATE_OF_BIRTH'] = pd.to_datetime(df['DATE_OF_BIRTH'])
df['DISBURSAL_DATE'] = pd.to_datetime(df['DISBURSAL_DATE'])

# The columns 'AVERAGE_ACCT_AGE' and 'CREDIT_HISTORY_LENGTH' has data stored like "1year 10 months". Let's convert it.
# loop through the two columns and apply the conversion defined on FRTPFunctions file
for col in ['AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH']:
    df[col] = df[col].apply(lambda x:func.convertPeriod(x)) 
    
# list of columns to drop
toDrop = ["SUPPLIER_ID", "CURRENT_PINCODE_ID", "EMPLOYEE_CODE_ID", "PERFORM_CNS_SCORE"]

# Drop columns from the list and store on a new DataFrame df_slim
df_slim = df.drop(labels=toDrop, axis=1)

# For columns "DATE_OF_BIRTH" and "DISBURSAL_DATE", replace a datetime with the number of years passed since the min(date) of col, in float format
for col in ["DATE_OF_BIRTH" , "DISBURSAL_DATE"]:
    func.convertToYears(df_slim, col)
    
# call the function to split into train and test and removes more columns
X_train, X_test, y_train, y_test = func.prepDataSlimmer(df_slim)

# Define ml
clf = make_pipeline(StandardScaler(), SGDClassifier(class_weight='balanced', random_state=42)).fit(X_train, y_train)
y_pred = clf.predict(X_test)
func.printData(clf, X_test, y_test)