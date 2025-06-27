from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

x=np.array([[30, 2, 5, 500], [50, 3, 10, 300], [45, 3, 8, 400], [25, 1, 2, 600], [60, 4, 15, 200],
    [55, 3, 12, 250], [40, 2, 7, 450], [35, 2, 6, 480], [48, 3, 9, 350], [28, 1, 3, 550],
    [38, 2, 5, 520], [52, 3, 11, 280], [42, 2, 7, 410], [32, 1, 4, 530], [58, 4, 14, 230]])
y=np.array([8, 15, 13, 6, 20, 17, 11, 10, 14, 7, 9, 16, 12, 7, 19])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3)

model=LinearRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print("mse準確率:", mse)
print("r2", r2)

a=float(input("請輸入您的房屋面積"))
b=float(input("請輸入您想要的房間數:"))
c=float(input("請輸入您房屋的屋齡"))
d=float(input("請輸入距離捷運站距離"))

new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)
result=model.predict(now_data)[0]

print(f"預測房價:{result:.2f}百萬元")