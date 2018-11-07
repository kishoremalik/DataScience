# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:04:46 2018

@author: kishore Malik
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def Load_Data(path):
    data=pd.read_csv(path)
    return data


    
def missing_val(data):
    return data.isnull().sum()

def eda_describe(data):
    dec=data.describe()
    return dec 

def corelation_test(data):
    corr_data=data.corr()
    return corr_data 

## heat map
def heat_map(corr):
    sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns) 
        
    
    
def data_split(data):
    cols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'] 
    tar_cols='Outcome'
    x=data[cols]
    y=data[tar_cols]
    X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train,X_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X_train)
    scaler1=MinMaxScaler(feature_range=(0, 1))
    rescaledY=scaler1.fit_transform(X_test)
    return rescaledX,rescaledY

def model(X_scaled_train,y_train,X_scaled_test,y_test):
    clf = tree.DecisionTreeClassifier()
    print("x scaled=",X_scaled_train)
    clf.fit(X_scaled_train,y_train) 
    print("-------------------")
    pred=clf.predict(X_scaled_test)
    print("pred=",pred)
    accuracy=accuracy_score(y_test,pred)
    print("accuracy",accuracy)
    return pred,accuracy

def accuracy_matrix(pred,y_test):
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_test, pred, target_names=target_names))
    
    



