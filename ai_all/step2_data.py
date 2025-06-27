import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 讀取 CSV
df = pd.read_csv("vehicle_data.csv")

# 特徵與標籤
X = df[["avg_speed", "acc_var"]]
y = df["label"]

# 切分訓練 / 測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 測試準確率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ 測試準確率:", round(accuracy * 100, 2), "%")

# 測試預測一筆新資料
test_speed = 40   # 自訂平均速度
test_acc_var = 2.5  # 自訂加速度變化
prediction = model.predict([[test_speed, test_acc_var]])
print(f"📍 預測你現在是：{prediction[0]}")
