import numpy as np
from sklearn.preprocessing import  StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

x=np.array([[25, 6.5, 120, 25], [30, 6.8, 110, 23], [35, 7.0, 130, 27], [40, 6.2, 100, 22],
    [28, 6.7, 115, 24], [45, 6.9, 125, 26], [32, 7.2, 140, 28], [29, 6.6, 118, 24],
    [36, 7.1, 135, 27], [33, 6.5, 112, 23], [38, 6.3, 108, 22], [26, 6.4, 109, 21],
    [27, 6.7, 122, 25], [31, 7.0, 130, 26], [34, 6.6, 115, 24]])
y=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

y=to_categorical(y)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(4, )),
    Dense(8, activation='relu'),
    Dense(3, activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop=EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=1)

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.4f}")

y_pred=model.predict(x_test)

y_testz=np.argmax(y_test, axis=1)
y_predz=np.argmax(y_pred, axis=1)

print(confusion_matrix(y_testz, y_predz))
print(classification_report(y_testz, y_predz))