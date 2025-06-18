import numpy as np
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix



(x, y), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y = y.flatten()
y_test = y_test.flatten()

x = x.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

y_train = to_categorical(y_train, num_classes=100)
y_val = to_categorical(y_val, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(200, activation='relu'),
    Dropout(0.3),
    Dense(150, activation='relu'),
    Dropout(0.2),
    Dense(100, activation='softmax')
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_val, y_val), 
          callbacks=[early_stop])

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

y_pred = model.predict(x_test)
y_testz = np.argmax(y_test, axis=1)
y_predz = np.argmax(y_pred, axis=1)

print(confusion_matrix(y_testz, y_predz)) 