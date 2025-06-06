from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

x=np.array([[90, 85, 78, 88], [75, 70, 65, 72], [60, 55, 50, 58],
    [95, 90, 85, 92], [80, 78, 70, 75], [50, 45, 40, 42],
    [88, 82, 77, 85], [70, 60, 58, 65], [40, 30, 25, 35],
    [92, 88, 84, 90], [78, 75, 70, 76], [65, 60, 55, 62]])
y=np.array([2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 1])

y=to_categorical(y)

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(4, )),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=8, verbose=1)

loss, acc=model.evaluate(x_test, y_test)
print(f"準確率:{acc:.2f}")