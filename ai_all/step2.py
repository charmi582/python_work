import numpy as np
from keras.models import load_model

# 載入模型與標籤
model = load_model("vehicle_classifier_model.h5")
label_classes = np.load("vehicle_label_classes.npy", allow_pickle=True)

def predict_vehicle(avg_speed, acc_var):
    X_input = np.array([[avg_speed, acc_var]])
    y_pred = model.predict(X_input)
    label_idx = np.argmax(y_pred)
    return label_classes[label_idx]

# 範例
result = predict_vehicle(28, 4.7)
print(f"自動判斷結果：{result}")
