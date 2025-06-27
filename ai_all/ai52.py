import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical 
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten
from keras.layers import RandomRotation, RandomZoom, RandomContrast, RandomFlip, RandomBrightness
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from keras.datasets import cifar100
from keras.regularizers import l2

(x, y), (x_test, y_test)=cifar100.load_data()

y=y.flatten()
y_test=y_test.flatten()

x=x.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

data=Sequential([
    Input(shape=(32, 32, 3)),
    RandomFlip('horizomtal'),
    RandomContrast(0.1),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomBrightness(0.1)
])

model=Sequential([
    data,
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.1)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(100,  activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce=ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, epochs=50, batch_size=200, verbose=1, validation_data=(x_val, y_val),
          callbacks=[early_stop, reduce])

loss, acc=model.evaluate(x_test, y_test)

print(f"準確率:{acc:.4f}")

y_pred=model.predict(x_test)

y_test=np.argmax(y_test, axis=1)
y_pred=np.argmax(y_pred, axis=1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))