# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:27:28 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def data_load(boston):
    data=pd.DataFrame(boston.data)
    data.columns=boston.feature_names
    data['price']=boston.target
    
    return data

""" scatter plot """
def scatter_plot(data):
    sns.regplot(data.CRIM, data.price)
    plt.xlabel("Per capita crime rate by town (CRIM)")
    plt.ylabel("Housing Price")
    plt.title("Relationship between CRIM and Price")
    
def scatter_plot1(data): 
    plt.scatter(data.RM, data.price)
    plt.xlabel("Average number of rooms per dwelling(RM)")
    plt.ylabel("Housing Price")
    plt.title("Relationship between RM and Price")
    
def corelation_matrix(bos):
    sns.set(style="white")

    df_corr= bos[:]
    # Compute the correlation matrix
    corr = df_corr.dropna().corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, ax=ax)
    
# train test split
def split(data):
    X=data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
    y=data.price
    X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

# model
def model(X_train,y_train,X_test,y_test):
    lm = LinearRegression()
    
    lm.fit(X_train, y_train)
    pred=lm.predict(X_test)
    
    coef=lm.coef_
    mse=np.mean(np.square(pred - y_test))
    print("mean squared error=",mse)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % lm.score(X_test, y_test))
    
    
    
    
    
    