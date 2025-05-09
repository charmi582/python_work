# traffic_classifier.py
import numpy as np
from tensorflow.keras.models import load_model

# 載入模型與標籤
model = load_model("model/vehicle_classifier_model_v2.h5")
label_classes = np.load("model/vehicle_label_classes_v2.npy", allow_pickle=True)

def predict_vehicle(avg_speed, acc_var):
    input_data = np.array([[avg_speed, acc_var]])
    prediction = model.predict(input_data, verbose=0)
    index = np.argmax(prediction)
    return label_classes[index]
