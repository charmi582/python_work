from keras.layers import Input, Flatten, Dense, MaxPool2D
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Add
from keras.applications import resnet50


def resnet_block(x, filters, kernel_size=3):
    shortcut = x  # skip connection

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])  # 把殘差加回來
    x = Activation('relu')(x)
    return x
inputs = Input(shape=(28, 28, 1))  # MNIST 一張圖
x = Conv2D(32, 3, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPool2D(2)(x)

x = resnet_block(x, 32)
x = resnet_block(x, 32)

x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
