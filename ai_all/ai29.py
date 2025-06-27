from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np

iris=load_iris()
x=iris.data
y=to_categorical(iris.target)

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3)

model=Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=8, verbose=1)

loss, accuracy=model.evaluate(x_test, y_test)

print(f"測試準確率:{accuracy:.2f}")
