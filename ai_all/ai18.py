from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

x=np.array([[25, 3, 600, 10], [45, 8, 750, 20], [30, 5, 700, 15], [22, 2, 550, 5], [35, 7, 720, 18],
    [29, 4, 680, 12], [50, 9, 780, 25], [27, 3.5, 610, 9], [40, 6.5, 740, 19], [32, 5.5, 690, 13],
    [28, 4.2, 670, 11], [48, 9.5, 770, 23], [34, 6, 710, 16], [23, 2.5, 580, 7], [41, 7.5, 730, 21],
    [36, 7, 720, 17], [26, 3.8, 660, 10], [44, 8.2, 750, 22], [31, 5, 700, 14], [49, 9.8, 780, 24]])

y=np.array([0,1,1,0,1,0,1,0,1,1,0,1,1,0,1,1,0,1,1,1])


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, stratify=y)

scaler=StandardScaler()
x_scaler_train=scaler.fit_transform(x_train)
x_scaler_test=scaler.fit_transform(x_test)

model=LogisticRegression()
model.fit(x_scaler_train, y_train)
score=cross_val_score(model, x_scaler_train, y_train, cv=5)
print("score準確率:", score)

y_pred=model.predict(x_scaler_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:", acc)

a=float(input("年齡（歲）"))
b=float(input("月收入（萬元）"))
c=float(input("信用評分（分數）"))
d=float(input("貸款金額（萬元）"))
new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)
result=model.predict(now_data)[0]
print("可以借貸"if result==1 else "不可以借貸")


