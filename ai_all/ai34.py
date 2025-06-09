from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import image_dataset_from_directory
import numpy as np

load_dir = r'C:\Users\charmi\Desktop\flower_photos'

dataset = image_dataset_from_directory(
    load_dir,
    image_size=(28, 28),
    color_mode='grayscale',
    label_mode='int',
    batch_size=None
)

x = []
y = []

for img, label in dataset:
    x.append(img.numpy())
    y.append(label.numpy())

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.8),
    Dense(32, activation='relu'),
    Dropout(0.8),
    Dense(16, activation='relu'),
    Dropout(0.8),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=5, verbose=1)

loss, acc = model.evaluate(x_test, y_test)
print(f"準確率: {acc:.4f}")
