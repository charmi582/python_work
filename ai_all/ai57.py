import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import RandomBrightness, RandomContrast, RandomZoom, RandomRotation, RandomFlip
from keras.utils import to_categorical
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x, y), (x_test, y_test)=cifar10.load_data()

import tensorflow as tf
x=tf.image.resize(x, (64, 64))
x_test=tf.image.resize(x_test, (64, 64))

x=preprocess_input(x).numpy()
x_test=preprocess_input(x_test).numpy()

y=to_categorical(y, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.3, random_state=42, 
                                                stratify=y.argmax(axis=1))

base_model=ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
base_model.trainable=False


model=Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce=ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, epochs=25, batch_size=100, validation_data=(x_val, y_val), 
          callbacks=[early_stop, reduce], verbose=1)

model.evaluate(x_test, y_test)

print(f"準確率:")