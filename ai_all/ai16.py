from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

x=np.array([[6, 4, 1], [5, 5, 0.5], [7, 3, 1.5], [4, 6.5, 0], [6.5, 4.5, 1],
    [8, 2, 2], [3.5, 7, 0], [7.5, 3, 1.8], [6, 5, 0.2], [5.5, 5.5, 0],
    [6.8, 3.5, 1.2], [4.8, 6.8, 0.1]])
y=np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:", acc)

a=float(input("請輸入睡眠時間:"))
b=float(input("請輸入上課時間:"))
c=float(input("請輸入運動時間:"))
new_data=[[a, b, c]]
result=model.predict(new_data)[0]
print("您會及格" if result==0 else "您會不及格")