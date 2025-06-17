import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPool2D, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

(x, y), (x_test, y_test)=cifar10.load_data()

y=y.flatten()
y_test=y_test.flatten()

x=x.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)    

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_val=to_categorical(y_val)

model=Sequential([
    Conv2D(32,(3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(3, 3),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2), 
    Dense(25, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=1, callbacks=[early_stop],
           validation_data=(x_val, y_val))

loss, acc=model.evaluate(x_test, y_test)
print(f"準確率:{acc:.4f}")