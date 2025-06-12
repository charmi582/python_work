import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from sklearn.metrics import classification_report

x=np.array([[1, 0.5, 2, 2], [2, 1.0, 3, 4], [3, 1.5, 4, 6],
    [4, 2.0, 5, 12], [5, 2.5, 6, 18], [6, 3.0, 7, 24],
    [7, 3.5, 8, 30], [8, 4.0, 9, 36], [5, 2.5, 5, 20],
    [2, 1.0, 3, 5], [6, 3.0, 8, 28], [1, 0.5, 2, 1]])
y=np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0, 2, 0])

y=to_categorical(y)

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(4, )),
    Dense(8, activation='tanh'),
    Dense(3, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop=EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop])

loss, acc=model.evaluate(x_test, y_test)

print("準確率:", acc)

y_pred=model.predict(x_test)

y_testz=np.argmax(y_test, axis=1)
y_predz=np.argmax(y_pred, axis=1)

print(confusion_matrix(y_testz, y_predz))
print(classification_report(y_testz, y_predz))