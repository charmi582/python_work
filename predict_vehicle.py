from tensorflow.keras.models import load_model
import numpy as np

# 載入模型和標籤
model = load_model("vehicle_classifier_model.h5")
label_classes = np.load("vehicle_label_classes.npy", allow_pickle=True)

# 傳入平均速度與加速度變異，輸出交通工具類別
def predict_vehicle_type(speed, acc_var):
    input_data = np.array([[speed, acc_var]])
    prediction = model.predict(input_data)
    class_idx = np.argmax(prediction)
    return label_classes[class_idx]
