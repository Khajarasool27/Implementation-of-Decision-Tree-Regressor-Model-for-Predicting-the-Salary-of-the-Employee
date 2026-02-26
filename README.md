# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: B.Khaja Rasool
RegisterNumber: 212224230040
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## Data Head:

<img width="393" height="165" alt="Screenshot 2026-02-10 153604" src="https://github.com/user-attachments/assets/0e930414-57e6-478a-85c7-28e8811508bb" />


## Data Info:
<img width="391" height="263" alt="Screenshot 2026-02-10 153610" src="https://github.com/user-attachments/assets/96c343ee-22d7-416f-b4c7-ab67aff538e6" />


## isnull() sum():
<img width="209" height="125" alt="Screenshot 2026-02-10 153616" src="https://github.com/user-attachments/assets/39d4b654-b8f0-47de-90c3-19f3d8013f89" />


## Data head after Encoding:
<img width="304" height="161" alt="Screenshot 2026-02-10 153621" src="https://github.com/user-attachments/assets/8a1d7a95-8a1f-41c5-bc5f-0effbd5c0538" />



<img width="186" height="53" alt="Screenshot 2026-02-10 153626" src="https://github.com/user-attachments/assets/211aca3b-ca01-4530-a3f7-9b22e4fd08fc" />



<img width="244" height="62" alt="Screenshot 2026-02-10 153632" src="https://github.com/user-attachments/assets/56dc34ab-93e8-4829-9734-460f85c81f52" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
