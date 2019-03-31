# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 01:40:15 2019

@author: nakul
"""

import numpy as np
import pandas as pd
from AdvancedAnalytics import ReplaceImputeEncode
rie = ReplaceImputeEncode()
adf = df

data_map = rie.draft_data_map(adf)
adf_dropped = adf.dropna(subset = ['object'])
rie = ReplaceImputeEncode(data_map = data_map, drop = True, display = True)
adf_imputed = rie.fit_transform(adf_dropped)
X = np.asarray(adf_imputed.drop('object', axis = 1))
Y = np.asarray(adf_imputed['object'])
df = pd.read_excel('sonar3by5.xlsx')
df.head()
df.max()
df.min()
df.shape
df.isnull().sum()
dfg = (df > 1) | (df < 0)
dfg.sum()
df[df['R6']>1]
df[df.R6 > 1] = df.R6.mean()
df[df.R5 > 1] = df.R5.mean()
df[df.R19 > 1] = df.R19.mean()
df.max()
df.columns
X = df.drop(['object'], axis = 1)
Y = df['object']
Y.iloc[12] = 'R'
Y.iloc[19] = 'R'
Y.iloc[200] = 'M'
#Removing NA Values
#Take care of Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0) 
imputer = imputer.fit(X)
X = imputer.transform(X)

#Converting Categorical data
#Encoding categorical data using dummy variables for X
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(Y).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

#Predicting Test Set Results
y_pred = classifier.predict(X_test)
#Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
aol = pd.DataFrame([y_pred,Y_test])
aol.transpose()












