from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np

x=np.array([[5, 30, 50, 12], [2, 15, 30, 9], [6, 40, 60, 13],
    [1, 10, 25, 8], [3, 20, 35, 11], [7, 50, 70, 14],
    [2, 10, 28, 10], [4, 25, 45, 12], [6, 45, 65, 15]])
y=np.array([1, 0, 2, 0, 1, 2, 0, 1, 2])

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

model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=1)

loss, acc=model.evaluate(x_test, y_test)

y_pred=model.predict(x_test)
y_predz=np.argmax(y_pred, axis=1)
y_testz=np.argmax(y_test, axis=1)

print(f"準確率:{acc:.2f}")
print(classification_report(y_testz, y_predz))