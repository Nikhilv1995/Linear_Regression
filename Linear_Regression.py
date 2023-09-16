# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:49:37 2023

@author: nikhilve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the CSV data
data = pd.read_csv('Salary_Data.csv')


#selecting features and result.   OR Vector of DV(Dependent Variables) y, and Matrix of IV(Independent Variables) x
x=data.iloc[:,:-1].values

y=data.iloc[:,1:].values


#Using train-test split to break the data into training and testing data. test_size= 20%data is reserved for testing  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3, random_state=0)


#Importing Linear Regression, regressor will be our model.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Training the regressor.
regressor.fit(x_train, y_train)

print('This is the W1 value :',regressor.coef_)
print('This is the W0 value :',regressor.intercept_)
print("Predicted Salary :",regressor.predict([[9]]))

#predicting Output:
y_predict = regressor.predict(x_test)


#Visual representation of the training-set.
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.xlabel("Experience in Years :")
plt.ylabel("Salary")
plt.title("Training-Set Graph Salary vs Exp.")
plt.show()

# Visual representation of test-set.
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.xlabel("Experience in Years :")
plt.ylabel("Salary")
plt.title("Test-Set Graph Salary vs Exp.")
plt.show()




















































