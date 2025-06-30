import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Dropout, BatchNormalization, MaxPool2D, Flatten
from keras.layers import RandomBrightness, RandomRotation, RandomContrast, RandomFlip, RandomZoom
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import Input 

(x, y), (x_test, y_test)=mnist.load_data()

import tensorflow as tf
x=x.reshape(-1, 28, 28, 1)
x_test=x_test.reshape(-1, 28, 28, 1)

x=x.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

y=to_categorical(y)
y_test=to_categorical(y_test)

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.2, random_state=42,
                                                 stratify=y.argmax(axis=1))

model=Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce=ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, epochs=20, batch_size=256, verbose=1, validation_data=(x_val, y_val), 
          callbacks=[early_stop, reduce])

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.4f}")

y_pred=model.predict(x_test)
y_test=np.argmax(y_test, axis=1)
y_pred=np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))