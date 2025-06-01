from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

x=np.array([[25, 40, 650, 1.5], [35, 65, 720, 2.5], [45, 85, 780, 3.5], [22, 30, 600, 1], [40, 75, 750, 3],
    [30, 50, 700, 2], [50, 95, 800, 4], [28, 45, 680, 1.8], [38, 70, 740, 3], [33, 55, 710, 2.2],
    [29, 48, 690, 1.7], [47, 90, 790, 3.8], [36, 68, 730, 2.7], [24, 35, 640, 1.3], [44, 80, 770, 3.3],
    [31, 52, 705, 2], [27, 42, 675, 1.6], [41, 72, 755, 3], [34, 58, 715, 2.5], [49, 88, 785, 3.6]])
y=np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

des_model=DecisionTreeClassifier()
des_model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
con=confusion_matrix(y_test, y_pred)
print("acc準確率:", acc)
print("con準確率:", con)

knn_pred=knn_model.predict(x_test)
knn_acc=accuracy_score(y_test, knn_pred)
knn_con=confusion_matrix(y_test, knn_pred)
print("knn_acc準確率:", knn_acc)
print("knn_con準確率:", knn_con)

des_pred=des_model.predict(x_test)
des_acc=accuracy_score(y_test, des_pred)
des_con=confusion_matrix(y_test, des_pred)
print("des_acc準確率:", des_acc)
print("des_con準確率:", des_con)

a=float(input("年齡（歲）:"))
b=float(input("年收入（萬元）:"))
c=float(input("信用評分（300~850）:"))
d=float(input("每月房貸支出（萬元）:"))

new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)
result=model.predict(now_data)[0]
print("acc_會借貸" if result==1 else "acc_不會借貸")
result1=knn_model.predict(now_data)[0]
print("knn_會借貸" if result1==1 else "knn_不會借貸")
result2=des_model.predict(now_data)[0]
print("des_會借貸" if result2==1 else "des_不會借貸")