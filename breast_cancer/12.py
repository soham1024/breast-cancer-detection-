import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.svm import SVC


data=pd.read_csv('data.csv')

data.drop(['Unnamed: 32', 'id'], 1,inplace= True)
#print(data.isna().sum())

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis = 1)

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) 

model=SVC(kernel='linear')
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

