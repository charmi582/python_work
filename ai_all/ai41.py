import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

x=np.array([ [170, 70, 25, 45, 1], [165, 60, 20, 35, 3], [180, 85, 28, 50, 0],
    [172, 75, 30, 40, 0], [160, 50, 18, 28, 5], [175, 78, 27, 38, 1],
    [168, 68, 22, 32, 2], [185, 90, 29, 55, 0], [158, 48, 15, 25, 6],
    [162, 52, 20, 30, 4], [177, 80, 26, 43, 1], [166, 58, 19, 34, 3],
    [178, 83, 28, 46, 0], [163, 54, 17, 27, 5], [176, 79, 24, 41, 2]])
y=np.array([1, 2, 0, 0, 2, 1, 1, 0, 2, 2, 1, 2, 0, 2, 1])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

y=to_categorical(y)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(5, )),
    Dense(6, activation="tanh"),
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