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
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thaksha Rishi
RegisterNumber:  212223100058

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
```

## Output:

<img width="159" height="213" alt="image" src="https://github.com/user-attachments/assets/b9d6f3ba-c2b6-4c8a-b65b-41ca19343482" />


```
df.tail()
```

## Output:

<img width="161" height="206" alt="image" src="https://github.com/user-attachments/assets/8c264026-01ec-42f3-9b55-e7d441168ce5" />

```
x=df.iloc[:,:-1].values
x
```

## Output:

<img width="149" height="489" alt="image" src="https://github.com/user-attachments/assets/1a1131ad-a8e8-4d41-b106-a4cbd6a338ca" />

```
y=df.iloc[:,1].values
y
```

## Output:

<img width="659" height="47" alt="image" src="https://github.com/user-attachments/assets/4649559e-1c97-4e7e-971c-96e8411dc83d" />

```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```

## Output:

<img width="634" height="37" alt="image" src="https://github.com/user-attachments/assets/44d629e9-f02e-4477-923b-0fe275da9621" />

```
Y_test
```

## Output:

<img width="403" height="26" alt="image" src="https://github.com/user-attachments/assets/d2711770-8007-4b50-9616-a6fe43f9c792" />

```
print("Register Number: 212223100058")
print("Name: Thaksha Rishi")
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Register Number: 212223100058")
print("Name: Thaksha Rishi")
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,Y_pred,color="red")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

print("Register Number: 212223100058")
print("Name: Thaksha Rishi")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

# Output

<img width="667" height="546" alt="image" src="https://github.com/user-attachments/assets/87ae5412-2b9b-4e73-8813-e7b0f27184b0" />

<img width="706" height="552" alt="image" src="https://github.com/user-attachments/assets/e32659a2-5e61-4549-936e-36c213971827" />

<img width="309" height="126" alt="image" src="https://github.com/user-attachments/assets/be1bc462-f824-4604-bc7f-d7280fd4b143" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
