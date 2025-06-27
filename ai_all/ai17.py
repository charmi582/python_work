from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score

x=np.array([ [25, 35, 2], [40, 60, 3], [30, 50, 1], [22, 28, 0], [35, 45, 2],
    [29, 40, 1], [50, 80, 4], [27, 38, 2], [45, 75, 3], [32, 55, 1],
    [26, 33, 1], [48, 70, 4], [34, 52, 2], [23, 30, 1], [41, 65, 3]])

y=np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1])

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, stratify=y)
scaler=StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)

model=LogisticRegression()
model.fit(x_train_scaler, y_train)
score=cross_val_score(model, x_train_scaler, y_train, cv=5)
print("cross準確率:", score)

y_pred=model.predict(x_test_scaler)
acc=accuracy_score(y_test, y_pred)
print("準確率:", acc)

a=float(input("請輸入您的收入:"))
b=float(input("請輸入您的年齡:"))
c=float(input("請輸入您上禮拜購買的數量:"))
new_data=np.array([[b, a, c]])
now_data=scaler.transform(new_data)

result=model.predict(now_data)[0]
print("您會購買"if result==1 else "您不會購買")
