import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.callbacks import EarlyStopping

x=np.array([ [25, 1.75, 70, 3, 2],  [30, 1.80, 85, 2, 3], [22, 1.65, 55, 4, 1],
    [35, 1.78, 90, 1, 3],  [28, 1.70, 60, 5, 1], [40, 1.85, 95, 1, 3],
    [24, 1.68, 58, 4, 2],  [33, 1.76, 80, 2, 3], [26, 1.72, 62, 5, 1],
    [38, 1.82, 88, 1, 3],  [29, 1.74, 75, 3, 2], [21, 1.60, 50, 5, 1],
    [34, 1.80, 92, 2, 3],  [27, 1.69, 65, 4, 2], [36, 1.77, 85, 1, 3]])

y=np.array([0, 1, 0, 2, 0, 2, 0, 1, 0, 2, 1, 0, 2, 1, 2])

y=to_categorical(y)

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, random_state=42, stratify=y)

model=Sequential([
    Dense(10, activation='tanh', input_shape=(5, )),
    Dropout(0.2),
    Dense(6, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop=EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=4, verbose=1, callbacks=[early_stop])

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.2f}")

y_pred=model.predict(x_test)

y_testz=np.argmax(y_test, axis=1)
y_predz=np.argmax(y_pred, axis=1)

print(classification_report(y_testz, y_predz))