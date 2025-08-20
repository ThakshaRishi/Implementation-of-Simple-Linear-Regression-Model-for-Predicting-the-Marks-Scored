# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thaksha Rishi
RegisterNumber: 212223100058


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
  
*/
```

## Output:
## head
<img width="695" height="236" alt="image" src="https://github.com/user-attachments/assets/6c1a76cc-38c5-403c-98b9-a8c913d3333d" />
## tail
<img width="325" height="235" alt="image" src="https://github.com/user-attachments/assets/926962bf-2d42-417e-9ce4-d61af7f8be54" />

## X
<img width="609" height="505" alt="image" src="https://github.com/user-attachments/assets/0be626bd-d8c6-433e-9cca-f06e1456397a" />

## Y

<img width="622" height="122" alt="image" src="https://github.com/user-attachments/assets/5e9ff43b-452f-45c2-8e1c-c0bf314fe691" />


## Y_pred


<img width="620" height="69" alt="image" src="https://github.com/user-attachments/assets/9b4482ea-bd80-4d66-a485-936e45cbc30f" />



## Y_test


<img width="495" height="32" alt="image" src="https://github.com/user-attachments/assets/19a1c809-9479-4001-8ca1-62940fec0f2c" />

## training set
<img width="827" height="712" alt="image" src="https://github.com/user-attachments/assets/5331e0d5-6ede-462d-98e8-fe048482cbce" />

## testing set


<img width="838" height="700" alt="image" src="https://github.com/user-attachments/assets/c1d0e3bc-8ad9-409c-aef0-1fe1f0cdfd2b" />


<img width="898" height="426" alt="image" src="https://github.com/user-attachments/assets/4660b122-2a4a-4084-b9e0-b9ef69f7f91d" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
