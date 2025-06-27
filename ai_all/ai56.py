import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import RandomBrightness, RandomRotation, RandomZoom, RandomContrast, RandomFlip
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report

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

data=Sequential([
    RandomFlip('horizontal'),
    RandomBrightness(0.1),
    RandomContrast(0.1),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

model=Sequential([
    base_model,
    data,
    GlobalAveragePooling2D(),
    Dense(25, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce=ReduceLROnPlateau(monitor='val_accuracy', patience=3,factor=0.5 ,verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, epochs=15, batch_size=100, verbose=1, callbacks=[early_stop, reduce],
          validation_data=(x_val, y_val))

loss, acc=model. evaluate(x_test, y_test)

print(f"準確率:{acc:.4f}")

y_pred=model.predict(x_test)

y_test=np.argmax(y_test, axis=1)
y_pred=np.argmax(y_pred, axis=1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))