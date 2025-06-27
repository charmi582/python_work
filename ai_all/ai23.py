from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

x=np.array([[25, 2, 3, 2], [35, 5, 6, 3], [45, 10, 9, 4], [22, 1, 2, 1], [40, 7, 7, 3],
    [30, 3, 4, 2], [50, 12, 10, 5], [28, 2.5, 3, 2], [38, 6, 5, 3], [33, 4, 4, 2],
    [29, 3.5, 4, 2], [47, 11, 9, 4], [36, 5.5, 6, 3], [24, 1.5, 3, 1], [44, 9, 8, 4],
    [31, 4, 4, 2], [27, 2, 3, 2], [42, 8, 7, 3], [34, 4.5, 5, 3], [48, 11.5, 9, 4]])
y=np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
confusion=confusion_matrix(y_test, y_pred)
print("acc準確率:", acc)
print("confusion", confusion)

x_pred=model.predict(x_test)
knn_acc=accuracy_score(y_test, x_pred)
knn_confusion=confusion_matrix(y_test, x_pred)
print("knn_acc準確率:", acc)
print("knn_confufion", knn_confusion)

a=float(input("年齡（歲）"))
b=float(input("每月消費金額（萬元）"))
c=float(input("消費頻率（次/每月）"))
d=float(input("會員等級（1~5）"))
new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)
result=model.predict(now_data)[0]
result1=model.predict(now_data)[0]

print("預測回購:會"if result==1 else "預測回購:不會")
print("預測回購:會"if result1==1 else "預測回購:不會")