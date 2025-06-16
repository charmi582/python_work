from  keras.datasets import cifar100
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.preprocessing import image_dataset_from_directory
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from seaborn import heatmap

dir=cifar100()

data_set=image_dataset_from_directory(
    dir,
    image_size=(32, 32),
    color_mode='rgb',
     label_mode='int',
    batch_size=None,
)

x=[]
y=[]

for img, label in data_set:
    x.append(img.numpy())
    y.append(label.numpy())

x=np.array(x)
y=np.array(y)

x=x.astype('float32')/255.0
y=to_categorical(y)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, stratify=y)

model=Sequential([
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

model.compile(optimizer='Stachastic_Gradient_Descent', loss='Sparse_Categorical_crossentropy', 
              metrics=['accuracy'])
early_stop=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=200, batch_size=100, validation_split=0.2, callbacks=[early_stop])

loss, acc=model.evaluate(x_test, y_test)

y_pred=model.predict(x_test)

y_testz=np.argmax(y_test, axis=1)
y_predz=np.argmax(y_pred, axis=1)

print(confusion_matrix(y_testz, y_predz))
print(heatmap(y_testz, y_predz))
