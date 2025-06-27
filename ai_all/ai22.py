from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

x=np.array([[3, 40, 4, 50], [8, 50, 3, 80], [2, 45, 5, 55], [10, 55, 2, 90], [4, 42, 4, 60],
    [1, 38, 5, 48], [12, 60, 1, 100], [5, 44, 3, 65], [7, 48, 4, 75], [3, 41, 5, 52],
    [6, 46, 4, 70], [9, 52, 3, 85], [2, 39, 5, 50], [11, 58, 2, 95], [4, 43, 4, 63]])

y=np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
confusion=confusion_matrix(y_test, y_pred)
print("acc準確率:", acc)
print("confution準確率:", confusion)

a=float(input("請輸入工齡（年)"))
b=float(input("請輸入每週工時（小時）："))
c=float(input("請輸入年度評分（1~5）"))
d=float(input("請輸入年薪（萬元）"))

new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)
result=model.predict(now_data)[0]
print("預測結果:流失"if result==1 else "預測結果:不流失")