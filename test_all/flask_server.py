from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允許跨來源存取（重要！）

# 載入模型（簡單直接訓練）
df = pd.read_csv("vehicle_data.csv")
X = df[["avg_speed", "acc_var"]]
y = df["label"]
model = RandomForestClassifier()
model.fit(X, y)

@app.route("/api/gps", methods=["POST"])
def receive_gps():
    data = request.json
    speed = data.get("speed", 0.0)

    # 模擬加速度變異：低速時較劇烈
    if speed < 15:
        acc_var = 4.5
    elif speed < 35:
        acc_var = 2.5
    else:
        acc_var = 1.0

    # 預測交通工具
    input_df = pd.DataFrame([[speed, acc_var]], columns=["avg_speed", "acc_var"])
    prediction = model.predict(input_df)[0]

    # 計算碳排
    if prediction == "car":
        co2 = speed * 0.192
    else:
        co2 = speed * 0.073

    return jsonify({
        "prediction": prediction,
        "co2": round(co2, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
