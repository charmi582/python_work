import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import RandomBrightness, RandomRotation, RandomContrast, RandomFlip, RandomZoom
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPool2D , GlobalAveragePooling2D
from keras.datasets import mnist
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import to_categorical

(x, y), (x_test, y_test)=mnist.load_data()

import tensorflow as tf
x=x.reshape(-1, 28, 28, 1)
x_test=x_test.reshape(-1, 28, 28, 1)

x=preprocess_input(x)
x_test=preprocess_input(x_test)

y=to_categorical(y)
y_test=to_categorical(y_test)


x_train, y_train, x_val, y_val=train_test_split(x, y, test_size=0.2, random_state=42,
                                                stratify=y.argmax(axis=1))

base_model=ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
base_model.trainable=True

data=Sequential([
    
    RandomFlip('horizomtal'),
    RandomZoom(0.1),
    RandomBrightness(0.1),
    RandomRotation(0.1),
    RandomContrast(0.1)
])

model=Sequential([
    data,
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(100, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce=ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=1, validation_data=[x_val, y_val],
          callbacks=[early_stop, reduce])

base_model.trainable=True

for layers in base_model.layers[:-20]:
    base_model.trainable=False

model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=1,
          validation_data=[x_val, y_val], callbacks=[early_stop, reduce])

loss, acc=model.evaluate(x_test, y_test)
print(f"準確率{acc:.4f}")
