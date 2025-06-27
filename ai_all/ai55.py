import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from keras.layers import RandomZoom, RandomRotation, RandomContrast, RandomFlip, RandomBrightness
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x, y), (x_test, y_test)=cifar100.load_data()

import tensorflow as tf
x=tf.image.resize(x, [64, 64])
x_test=tf.image.resize(x_test, [64, 64])

x=preprocess_input(x).numpy()
x_test=preprocess_input(x_test).numpy()

y=to_categorical(y)
y_test=to_categorical(y_test)

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.3, random_state=42,
                                                 stratify=y.argmax(axis=1))

base_model=ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
base_model.trainable=False

data=Sequential([
    RandomFlip('horizomtal'),
    RandomContrast(0.1),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomBrightness(0.1)
])

model=Sequential([
    base_model,
    data,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(100, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce=ReduceLROnPlateau(monitor='valaccuracy', patience=3, factor=0.5, verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, epochs=50, batch_size=250  , verbose=1, validation_data=(x_val, y_val),
          callbacks=[early_stop, reduce])

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.4f}")