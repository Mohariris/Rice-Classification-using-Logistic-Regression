import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


columns = ["Area Integer", "Perimeter Real",
           "Major_Axis_Length Real",
           "Minor_Axis_Length Real", "Eccentricity	Real",
           "Convex_Area Integer", "Extent Real", "Cammeo_Osmancik"]
dataset = pd.read_csv("Rice_Cammeo_Osmancik.arff")
dataset.columns=columns

X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

#Encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#training the data
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=0)
print(X)

#Scaling the dataset
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#using logistic regression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predicting test set

y_pred=classifier.predict(X_test)

#checking accuracy score

cm=confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

