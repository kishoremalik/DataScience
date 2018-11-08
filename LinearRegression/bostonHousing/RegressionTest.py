# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:24:14 2018

@author: Kishore malik
"""

from sklearn.datasets import load_boston
import RegressionTestHelper as reg

boston = load_boston()

"""  data load """
boston_data=reg.data_load(boston)
print(boston_data.head()) 


print(boston_data.columns)


"""EDA and Summary Statistics """ 

print(boston_data.describe())

""" scatter plot """
#reg.scatter_plot(boston_data)

print("scatter plot for RM and price")
#reg.scatter_plot1(boston_data)

""" Corelation Matrix """
# RAD and TAX are highly co-related.
#Price negatively corelated with LSTAT(Strong),PTRATIO(Strong),TAX(high), INDUS(High), CRIM(Highly) and 
#NOX highly corelated with RM.
#Also Price positively corelated with RM(High), ZN(High), CHAS(Medium), DIS(MEDIUM) & B(Medium)

#reg.corelation_matrix(boston_data)

print(boston_data.columns)
X_train, X_test, y_train, y_test=reg.split(boston_data)

""" model build and error check """
reg.model(X_train,y_train,X_test,y_test)