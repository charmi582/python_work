from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

x=np.array([[25, 5, 120, 10], [45, 15, 40, 20], [30, 3, 300, 5], [50, 25, 20, 25],
    [22, 4, 180, 8], [38, 10, 60, 18], [28, 2, 330, 3], [60, 40, 10, 30],
    [33, 8, 150, 12], [41, 12, 70, 22], [26, 6, 100, 9], [55, 30, 25, 26],
    [29, 4, 250, 6], [48, 20, 50, 21], [31, 7, 140, 11], [36, 9, 90, 15],
    [24, 3, 270, 4], [52, 35, 18, 28], [27, 5, 160, 10], [43, 18, 45, 19]])
y=np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])

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
print(f"acc準確率:", acc)
print("混淆矩陣:", con)

x_pred=knn_model.predict(x_test)
acc1=accuracy_score(y_test, x_pred)
con1=confusion_matrix(y_test, x_pred)
print("knn_acc準確率:", acc1)
print("knn_混淆矩陣:", con1)

z_pred=des_model.predict(x_test)
acc2=accuracy_score(y_test, z_pred)
con2=confusion_matrix(y_test, z_pred)
print("des_acc準確率:", acc2)
print("des混淆矩陣", con2)

a=float(input("age（年齡）"))
b=float(input("total_spent（總消費金額，單位：萬元）"))
c=float(input("last_days（距上次購買天數）"))
d=float(input("items（總共購買過幾項商品）"))

new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)

result=model.predict(now_data)[0]
print("會再次購買"if result==1 else "不會再次購買")
result1=knn_model.predict(now_data)[0]
print("會再次購買"if result1==1 else "不會再次購買")
result2=des_model.predict(now_data)[0]
print("會再次購買"if result2==1 else "不會再次購買")