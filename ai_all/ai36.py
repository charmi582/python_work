from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.metrics import classification_report

x=np.array([ [25, 80, 3, 70], [30, 90, 4, 85], [28, 85, 2, 80], [22, 70, 1, 60], [35, 95, 5, 90],
    [27, 75, 2, 65], [40, 88, 6, 92], [32, 82, 4, 78], [29, 86, 3, 84], [24, 72, 1, 68],
    [26, 78, 2, 75], [34, 91, 5, 89], [31, 87, 3, 83], [23, 69, 1, 61], [36, 96, 6, 94]])
y=np.array([
    0, 1, 1, 0, 1,
    0, 2, 2, 2, 0,
    0, 1, 2, 0, 2
])

y=to_categorical(y)

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_scaler, y, test_size=0.3, stratify=y)

model=Sequential([
    Dense(10, activation='relu', input_shape=(4, )), 
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
early_stop=EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=2, verbose=1, validation_split=0.2, callbacks=[early_stop])

loss, acc=model.evaluate(x_test, y_test)

y_pred=model.predict(x_test)

y_testz=np.argmax(y_test, axis=1)
y_predz=np.argmax(y_pred, axis=1)
print(f"準確率:{acc}")
print(classification_report(y_testz, y_predz))