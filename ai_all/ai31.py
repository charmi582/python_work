from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Dense
from sklearn.metrics import classification_report
import numpy as np

x=np.array([[30, 60, 10, 35], [25, 55, 12, 30], [28, 58, 11, 32], [4, 35, 6, 20], [5, 38, 7, 18], [3.5, 32, 6, 19], [2, 30, 15, 25], [2.5, 33, 14, 23], [1.8, 29, 16, 22]])

y=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

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

model.save("frist_test.h5")
y_perd=model.predict(x_test)
y_perdz=np.argmax(y_perd, axis=1)
y_testz=np.argmax(y_test, axis=1)
print(classification_report(y_testz, y_perdz))
print(f"準確率:{acc:.2f}")