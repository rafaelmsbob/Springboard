# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:18:06 2020

@author: Rafael
"""

# Functions used on file "fromRawToPrediction"

# Import Libraries

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

def convertPeriod(str):
    tokens = re.findall("\d+", str)    
    return int(tokens[0]) + round(int(tokens[1])/12,2)


def convertToYears(df, col):
    # For a DataFrame df, this function replaces a datetime column col with the number of years passed since
    # the min(date) of col, in float format    
    minDate = min(df[col])    
    df[col] = df[col].apply(lambda x: (x - minDate)/np.timedelta64(1,'Y'))
    
def prepDataSlimmer(df):       
    # Identify column "LOAN_DEFAULT" as target
    if("LOAN_DEFAULT" in df.columns):
        y = df.LOAN_DEFAULT
        df = df.drop(labels="LOAN_DEFAULT", axis=1)
    
    # Remove columns ending in ID from the slim version (Branch ID, Manufacturer ID and State ID)
    # Create a list with columns ending in ID
    ID_cols = [col for col in df.columns if col[-2:] == "ID"]
        
    # Drop these columns
    df = df.drop(labels=ID_cols, axis=1)
    
    # List of columns to drop  -  for more noise reduction
    toDrop = ["LTV", "MOBILENO_AVL_FLAG", "AADHAR_FLAG", "PAN_FLAG", "VOTERID_FLAG", "DRIVING_FLAG", "PASSPORT_FLAG"]
    
    df = df.drop(labels=toDrop, axis=1)
    
    # Generate dummy columns on the categorical columns of df
    df = pd.get_dummies(df, drop_first=True)    
    
    # Split into trainig and testing
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=0, stratify=y, shuffle=True)
    
    return X_train, X_test, y_train, y_test

# Function that prints the relevant data about a model
def printData(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print("total predicted Defaults: ",sum(y_pred))
    print("total Defaults on test: ", sum(y_test))
    print("R Squared: ", accuracy_score(y_test, y_pred))
    print("F1Score: ", f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Recall: ", cm[1,1]/(cm[1,0]+cm[1,1]))
