from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

x=np.array([[85, 90, 5, 78], [92, 88, 7, 85], [75, 70, 2, 60], [65, 65, 1, 55], [80, 75, 4, 72],
    [95, 95, 8, 90], [55, 50, 1, 45], [78, 80, 3, 68], [88, 85, 6, 82], [70, 68, 2, 58],
    [82, 88, 4, 75], [90, 90, 7, 88], [60, 55, 1, 50], [85, 83, 5, 77], [73, 70, 2, 62],
    [97, 96, 9, 92], [68, 65, 2, 57], [80, 78, 4, 70], [76, 74, 3, 65], [89, 87, 6, 84]])
y=np.array([82, 90, 65, 55, 75, 95, 48, 70, 85, 60, 77, 92, 52, 80, 68, 97, 62, 72, 69, 88])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, random_state=42)

model=LinearRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
r2=r2_score(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
print("r2準確度:", r2)
print("mse準確度:", mse)

a=float(input("上課出席率（百分比）"))
b=float(input("平時作業成績（百分比）"))
c=float(input("每週自習時數（小時）"))
d=float(input("期中考成績（百分比）"))

new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)
result=model.predict(now_data)
print(f"您的預測分數為:{result}")