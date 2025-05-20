from flask import Flask, request
from keras.models import load_model
import numpy as np
import datetime
from firestore_handler import upload_to_firestore

app = Flask(__name__)

# 載入模型與標籤
model = load_model("vehicle_classifier_model_v2.h5")
labels = np.load("vehicle_label_classes_v2.npy")

# 各交通工具碳排係數（g/km）
emission_factors = {
    "car": 171,
    "motorbike": 72,
    "bike": 0,
    "walk": 0
}

@app.route('/upload', methods=['POST'])
def upload():
    lat = float(request.form['lat'])
    lng = float(request.form['lng'])
    speed = float(request.form['speed'])

    # 用速度進行預測（可加入其他特徵）
    prediction = model.predict(np.array([[speed]]), verbose=0)
    predicted_label = labels[np.argmax(prediction)]

    # 假設一次抓資料距離為 1 公里（未來可換成實際距離）
    distance_km = 1
    carbon = emission_factors.get(predicted_label, 0) * distance_km

    # 整理資料
    record = {
        "lat": lat,
        "lng": lng,
        "speed": speed,
        "vehicle_type": predicted_label,
        "carbon_emission": carbon
    }

    # 上傳到 Firebase
    upload_to_firestore(record)

    return f"✅ 已成功接收並分析：你是 {predicted_label}，碳排放：{carbon} g"

if __name__ == '__main__':
    app.run(debug=True)

