from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

knn_model=KNeighborsClassifier()
knn_model.fit(x_train, y_train)

des_model=DecisionTreeClassifier()
des_model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("acc準確率", acc)

x_pred=knn_model.predict(x_test)
knn_acc=accuracy_score(y_test, x_pred)
print("knn_acc準確率:", knn_acc)

z_pred=des_model.predict(x_test)
des_acc=accuracy_score(y_test, z_pred)
print("des準確率:", des_acc)

a=float(input("輸入花萼長度（cm）："))
b=float(input("輸入花萼寬度（cm）："))
c=float(input("輸入花瓣長度（cm）："))
d=float(input("輸入花瓣寬度（cm）："))

new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)

result=model.predict(now_data)[0]
print(f"acc{result}")
result1=knn_model.predict(now_data)[0]
print(f"knn{result1}")
result2=des_model.predict(now_data)[0]
print(f"des{result2}")