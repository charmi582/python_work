from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np


x=np.array([[4, 120, 980, 15], [6, 150, 1250, 13], [8, 180, 1600, 12], [4, 95, 900, 16], [6, 140, 1100, 14],
    [4, 100, 950, 15], [8, 200, 1800, 11], [4, 105, 980, 15], [6, 130, 1150, 13], [4, 110, 1000, 16],
    [4, 90, 870, 17], [8, 190, 1700, 11], [6, 135, 1200, 14], [4, 100, 950, 16], [8, 210, 1850, 10]])
y=np.array([35, 25, 15, 38, 28, 36, 13, 37, 27, 36, 40, 14, 26, 36, 12])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, random_state=42)

model=LinearRegression()
model.fit(x_train, y_train)

knn_model=KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x_train, y_train)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)  
print("mse準確率:", mse)
print("r2準確率:", r2)

x_pred=knn_model.predict(x_test)
knn_mse=mean_squared_error(y_test, x_pred)
knn_r2=r2_score(y_test, x_pred)
print("knn_mse準確率:", knn_mse)
print("r2準確率:", knn_r2)

a=float(input("輸入汽缸數"))
b=float(input("輸入馬力"))
c=float(input("輸入車重(kg):"))
d=float(input("輸入加速度"))

new_data=np.array([[a, b, c, d]])
now_data=scaler.transform(new_data)
result=model.predict(now_data)
result1=knn_model.predict(now_data)
print(f"線性回歸方式下的油耗預測結果為:{result}公升")
print(f"knn下的油耗預測為{result1}公升")