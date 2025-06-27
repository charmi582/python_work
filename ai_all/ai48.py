from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import RandomZoom, RandomFlip, RandomRotation, RandomContrast
from keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import Input
import numpy as np


(x, y), (x_test, y_test)=mnist.load_data()

y=y.flatten()
y_test=y_test.flatten()

x=x.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

x=x.reshape(-1, 28, 28, 1)
x_test=x_test.reshape(-1, 28, 28, 1)

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.3, random_state=32, stratify=y)

y_train=to_categorical(y_train, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)
y_val=to_categorical(y_val)

data=Sequential([
    Input(shape=(28, 28, 1)),
    RandomFlip('horizontal'),
    RandomContrast(0.1),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

model=Sequential([
    data,
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(100, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(50, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=1, callbacks=[early_stop],
           validation_data=(x_val, y_val),  )

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.2f}")