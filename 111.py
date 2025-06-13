import tensorflow as tf
print("TensorFlow 版本：", tf.__version__)
print("是否有 GPU：", tf.config.list_physical_devices('GPU'))