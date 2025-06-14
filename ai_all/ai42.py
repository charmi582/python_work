import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

x=np.array([[65, 60, 55, 85], [70, 65, 60, 90], [75, 70, 68, 95],
    [50, 55, 52, 60], [55, 50, 45, 70], [45, 48, 50, 65],
    [80, 85, 82, 98], [78, 82, 80, 95], [85, 88, 90, 99],
    [60, 58, 62, 75], [65, 64, 63, 80], [68, 66, 65, 88],
    [40, 42, 45, 55], [45, 44, 43, 60], [50, 49, 48, 70],
    [90, 92, 94, 100], [88, 89, 90, 99], [85, 87, 88, 98]])
y=np.array([
    1, 1, 1,
    0, 0, 0,
    2, 2, 2,
    1, 1, 1,
    0, 0, 0,
    2, 2, 2
])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

y=to_categorical(y)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(4, )),
    Dense(6, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop], validation_split=0.2)

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.4f}")

y_pred=model.predict(x_test)

y_testz=np.argmax(y_test, axis=1)
y_predz=np.argmax(y_pred, axis=1)

print(classification_report(y_testz, y_predz))
print(confusion_matrix(y_testz, y_predz))