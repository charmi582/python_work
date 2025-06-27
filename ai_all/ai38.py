import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

x=np.array([[60, 55, 80, 65], [75, 70, 90, 80], [90, 85, 95, 88],
    [58, 50, 75, 60], [80, 78, 92, 85], [95, 90, 98, 92],
    [50, 45, 60, 55], [85, 80, 94, 89], [65, 60, 85, 70],
    [72, 68, 88, 75], [55, 50, 70, 58], [92, 88, 96, 90]])
y=np.array([0, 1, 2, 0, 1, 2, 0, 2, 1, 1, 0, 2])

y=to_categorical(y)

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, random_state=42, stratify=y)

model=Sequential([
    Dense(10, activation='tanh', input_shape=(4, )),
    Dropout(0.2),
    Dense(6, activation='relu'),
    Dropout(0.2), 
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
early_stop=EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=5, verbose=1, callbacks=[early_stop])

loss, acc=model.evaluate(x_test, y_test)

print("準確率:", acc)
y_pred=model.predict(x_test)

y_predz=np.argmax(y_pred, axis=1)
y_testz=np.argmax(y_test, axis=1)

print(classification_report(y_testz, y_predz))