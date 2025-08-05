import numpy as np
import pandas as pd
#from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('E:\\ML datasets\\diabetes.csv')

#print(data.head())

x= data.iloc[:,:-1]
y= data["Outcome"]
x = pd.get_dummies(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3, random_state = 42)

model = LogisticRegression(penalty = 'l2', C = 0.01, solver = 'liblinear')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test,y_pred))

scores = cross_val_score(model, x, y, cv=5)
print("Cross-validated accuracy:", scores.mean())