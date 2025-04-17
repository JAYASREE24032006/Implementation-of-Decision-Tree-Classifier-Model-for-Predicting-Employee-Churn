# EX-8 : IMPLEMENTATION OF DECISIONO TREE CLASSIFIER MODEL FOR PREDICTING EMPLOYEE CHURN

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Load and explore the employee dataset using pandas.

2.Encode categorical data (like salary) using LabelEncoder.

3.Define features (X) and target (y), then split the data.

4.Train a Decision Tree Classifier using the training set.

5.Predict and evaluate accuracy, then use the model for new predictions.

## PROGRAM:
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## OUTPUT:
![image](https://github.com/user-attachments/assets/10e85822-de46-4dd1-8a29-d74d43cf2ebf)


## RESULT:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
