import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from sklearn.metrics import confusion_matrix, classification_report

x=np.array([[1.2, 15, 500, 10, 0.8],
    [12.5, 3, 1200, 4, 2.5],
    [20.0, 1, 1500, 2, 3.2],
    [1.5, 20, 600, 9, 0.6],
    [13.0, 2, 1100, 5, 2.0],
    [22.0, 1, 1600, 3, 3.0],
    [1.0, 18, 550, 11, 0.7],
    [14.5, 2, 1150, 4, 2.2],
    [18.0, 1, 1400, 3, 3.5]])
y=np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

y=to_categorical(y)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(5, )),
    Dense(6, activation='tanh'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=1, callbacks=[early_stop],
           validation_split=0.2)

loss, acc=model.evaluate(x_test, y_test)

print(f'準確率:{acc:.4f}')

y_pred=model.predict(x_test)

y_testz=np.argmax(y_test, axis=1)
y_predz=np.argmax(y_pred, axis=1)

print(classification_report(y_testz, y_predz))
print(confusion_matrix(y_testz, y_predz))