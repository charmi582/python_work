from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np 

x=np.array([[2, 38, 3, 35], [5, 45, 4, 50], [3, 40, 2, 38], [8, 50, 5,  65], [1, 35, 1, 32], [6, 48, 4, 55], [2, 37, 2, 36], [4, 42, 3, 45], [3, 41, 2, 39], [7, 46, 5, 60]])
y=np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
print(classification_report(y_test, y_pred))