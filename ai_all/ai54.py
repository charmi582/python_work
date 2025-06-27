from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.layers import RandomFlip, RandomZoom, RandomContrast, RandomRotation, RandomBrightness
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.datasets import cifar10
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50

(x, y), (x_test, y_test)=cifar10.load_data()

import tensorflow as tf
x=tf.image.resize(x, [96, 96])
x_test=tf.image.resize(x, [96, 96])
 
x=preprocess_input(x).numpy()
x_test=preprocess_input(x_test).numpy()

y=to_categorical(y, num_classes=10)
y_test=to_categorical(y_test, num_classes=10)

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.3, random_state=42, 
                                                stratify=y.argmax(axis=1))

base_model=ResNet50(include_top=False, weights='imagenet', input_shape=(96, 96, 3))
base_model.trainable=False

model=Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(x_val, y_val))

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.4f}")