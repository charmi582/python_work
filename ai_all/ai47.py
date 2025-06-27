import numpy as np
from keras import Input
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
from keras.datasets import cifar100
from keras.layers import Dense, Conv2D, Normalization, Dropout, MaxPool2D, Flatten
from keras.layers import RandomContrast, RandomFlip, RandomRotation, RandomZoom
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


(x, y), (x_test, y_test)=cifar100.load_data()

y_test=y_test.flatten()
y=y.flatten()

x=x.astype('float32')/255.0
x_test.astype('float32')/255.0

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

y_test=to_categorical(y_test, num_classes=100)
y_train=to_categorical(y_train, num_classes=100)
y_val=to_categorical(y_val, num_classes=100)

data=Sequential([
    Input(shape=(32, 32, 3)),
    RandomFlip('horizontal'),
    RandomContrast(0.1), 
    RandomRotation(0.1),
    RandomZoom(0.1)
])

model=Sequential([
    data,
    Conv2D(32, (3, 3), activation='relu'),
    Normalization(),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Normalization(),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(200, activation='relu'),
    Normalization(),
    Dropout(0.2),
    Dense(150, activation="relu"),
    Normalization(), 
    Dropout(0.3),
    Dense(100, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=1, callbacks=[early_stop],
           validation_data=(x_val, y_val))

loss, acc=model.evaluate(x_test, y_test)

y_pred=model.predict(x_test)

y_test=np.argmax(y_test, axis=1)
y_pred=np.argmax(y_pred, axis=1)

print(f"準確率:{acc:.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))