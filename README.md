# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

data = data.drop(["sl_no", "salary"], axis=1)

# Placed = 1, Not Placed = 0
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

X = data.drop("status", axis=1)
y = data["status"]

X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
```

## Output:
<img width="725" height="311" alt="Screenshot 2026-02-14 081505" src="https://github.com/user-attachments/assets/71086f26-9929-4122-b1e6-c5d7b07a16ce" />
<img width="571" height="255" alt="Screenshot 2026-02-14 081636" src="https://github.com/user-attachments/assets/fc21b3ac-94b9-4d96-9953-3977d7774d42" />
<img width="491" height="304" alt="Screenshot 2026-02-14 081652" src="https://github.com/user-attachments/assets/32f01c06-9597-4c19-bb62-d91bb625dfa5" />
<img width="530" height="453" alt="ml-5" src="https://github.com/user-attachments/assets/48961342-4ac3-4f0b-a523-7ad83c8225b5" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
