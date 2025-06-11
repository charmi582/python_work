from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight


load_dir = r'C:\Users\charmi\Desktop\flower_photos'

dataset = image_dataset_from_directory(
    load_dir,
    image_size=(64, 64),
    color_mode='rgb',
    label_mode='int',
    batch_size=None,

)

x = []
y = []

for img, label in dataset:
    x.append(img.numpy())
    y.append(label.numpy())

x = np.array(x)
y = np.array(y)

x=x.astype('float32')/255.0
y=to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

y_train_raw = np.argmax(y_train, axis=1)

class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_raw),
    y=y_train_raw
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64 , 3)),
    BatchNormalization(), 
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(), 
    MaxPool2D(2, 2),
    Conv2D(128,(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)
class_weights_dict = dict(enumerate(class_weights_array))

model.fit(x_train, y_train, epochs=150, batch_size=100, verbose=1, callbacks=[early_stop],  class_weight=class_weights_dict)

loss, acc = model.evaluate(x_test, y_test)

y_pred=model.predict(x_test)
y_predz=np.argmax(y_pred, axis=1)
y_testz=np.argmax(y_test, axis=1)
print(f"準確率: {acc:.4f}")
print(confusion_matrix(y_testz, y_predz))
print(classification_report(y_testz, y_predz))
