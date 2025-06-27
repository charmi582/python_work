from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

x=np.array([[25, 3.5, 5, 2.0],
    [40, 6.0, 12, 3.5],
    [35, 5.2, 10, 3.0],
    [20, 2.5, 3, 1.2],
    [45, 7.5, 15, 4.0],
    [23, 3.0, 4, 1.8],
    [50, 8.0, 20, 5.0],
    [30, 4.0, 8, 2.5],
    [55, 9.0, 25, 5.5]])

y=np.array([0, 1, 1, 0, 2, 0, 2, 1, 2])

y=to_categorical(y)

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(4, )),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=2, verbose=1)

loss, acc=model.evaluate(x_test, y_test)

y_pred=model.predict(x_test)
y_predz=np.argmax(y_pred, axis=1)
y_testz=np.argmax(y_test, axis=1)

print(f"準確率:{acc}")
print(classification_report(y_testz, y_predz))