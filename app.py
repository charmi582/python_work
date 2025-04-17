import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# 1. 載入模型（如果你還沒存過，就用訓練好的模型）
@st.cache_resource
def load_model():
    # 或者直接在這裡重新訓練一次模型
    df = pd.read_csv("vehicle_data.csv")
    X = df[["avg_speed", "acc_var"]]
    y = df["label"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = load_model()

# --------------------------
# 2. Streamlit UI
st.title("🚗🛵 智慧交通碳排放計算器")
st.write("系統會根據你輸入的速度與加速度，自動判斷你是開車還是騎機車，並計算今日碳排放量")

# 輸入區
speed = st.slider("平均速度 (km/h)", 0.0, 100.0, 30.0)
acc = st.slider("加速度變化程度 (0~10)", 0.0, 10.0, 2.5)
km = st.number_input("今日行駛公里數", min_value=0.0, value=10.0)

# --------------------------
# 3. 模型預測
if st.button("計算碳排放"):
    input_df = pd.DataFrame([[speed, acc]], columns=["avg_speed", "acc_var"])
    prediction = model.predict(input_df)[0]
    
    if prediction == "car":
        co2 = km * 0.192
    else:
        co2 = km * 0.073

    st.success(f"你今天使用的是：🚙 {prediction}")
    st.info(f"今日碳排放總量為：**{round(co2, 2)} 公斤 CO₂**")
