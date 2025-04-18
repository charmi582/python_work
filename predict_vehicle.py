import numpy as np
from tensorflow.keras.models import load_model

# 載入模型與標籤名稱
model = load_model("vehicle_classifier_model.h5")
label_classes = np.load("vehicle_label_classes.npy", allow_pickle=True)

# 預測函式
def predict_vehicle_type(avg_speed, acc_var):
    input_data = np.array([[avg_speed, acc_var]])
    prediction = model.predict(input_data)
    label_index = np.argmax(prediction)
    return label_classes[label_index]

# 📦 範例：你可以直接測試看看
if __name__ == "__main__":
    speed = float(input("請輸入平均速度 (km/h): "))
    acc = float(input("請輸入加速度變異值: "))
    result = predict_vehicle_type(speed, acc)
    print(f"🚘 推論結果：你現在是 {result}")
