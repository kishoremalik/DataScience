# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:01:47 2018

@author: Kishore Malik

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import TreeHelper
from sklearn import tree


path="D:\Datasets\DiabaticData\diabetes.csv"
data=TreeHelper.Load_Data(path)
print(data.head())

## missing value calculation
missingVal=TreeHelper.missing_val(data)
#print("missing value=")
#print(missingVal) 

## EDA describe data
desc=TreeHelper.eda_describe(data)
#print("describe")
#print(desc) 

## corelation checking 
cor_data=TreeHelper.corelation_test(data)
#print("corelation of features") 
#print(cor_data)

## heat map 
#TreeHelper.heat_map(cor_data) 

cols=data.columns
#print(cols)

## train test data sets
X_train, X_test, y_train, y_test=TreeHelper.data_split(data)
#print(y_train)
# scaled data
X_scaled_train,X_scaled_test=TreeHelper.scale_data(X_train,X_test)
#print(X_scaled_test)

pred,accuracy=TreeHelper.model(X_scaled_train,y_train,X_scaled_test,y_test)
print("accuracy=",accuracy)

## f1 score, precesion , recall
TreeHelper.accuracy_matrix(pred,y_test)







