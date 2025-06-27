import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.layers import RandomBrightness, RandomContrast, RandomRotation, RandomZoom, RandomFlip
from keras.utils import to_categorical
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential

(x, y), (x_test, y_test)=cifar100.load_data()

x=tf.image.resize(x, (64, 64))
x_test=tf.image.resize(x_test, (64, 64))

x=preprocess_input(x).numpy()
x_test=preprocess_input(x_test).numpy()

y=to_categorical(y)
y_test=to_categorical(y_test)

x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.3, random_state=42,
                                                 stratify=y.argmax(axis=1))
base_model=ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
base_model.trainable=True

data=Sequential([
    RandomFlip('horizontal'),
    RandomBrightness(0.1),
    RandomContrast(0.1),
    RandomZoom(0.1),
    RandomRotation(0.1)
])

model=Sequential([
    data,
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(100, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce=ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=1, validation_data=(x_val, y_val),
          callbacks=[early_stop,reduce])

base_model.trainable=True

for layer in base_model.layers[:-30]:
     base_model.trainable=False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(
    x_train, y_train, epochs=10, batch_size=256, verbose=1,
    validation_data=(x_val, y_val), callbacks=[early_stop, reduce]
)

# 10. 測試準確率
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(f"✅ 測試集準確率: {acc:.4f}")