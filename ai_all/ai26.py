from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

califirnia=fetch_california_housing()
x=califirnia.data
y=califirnia.target

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, random_state=42)

model=LinearRegression()
model.fit(x_train, y_train)

knn_model=KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x_train, y_train)

des_model=DecisionTreeRegressor()
des_model.fit(x_train, y_train)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
print("mse準確率:", mse)

x_pred=knn_model.predict(x_test)
mse1=mean_squared_error(y_test, x_pred)
print("knn_mse準確率", mse1)

z_pred=des_model.predict(x_test)
mse2=mean_squared_error(y_test, z_pred)
print("des_mse準確率:", mse2)

a=float(input("區域住戶收入中位數"))
b=float(input("屋齡中位數"))
c=float(input("平均房間數"))
d=float(input("平均臥室數"))
e=float(input("人口數"))
f=float(input("平均居住人數"))
g=float(input("緯度"))
h=float(input("經度"))

now_data=np.array([[a, b, c, d, e, f, g, h]])
new_data=scaler.transform(now_data)
result=model.predict(new_data)[0]
print(f"房價:{result}")

result1=knn_model.predict(new_data)[0]
print(f"房價:{result1}")

result2=des_model.predict(new_data)
print(f"房價:{result2}")