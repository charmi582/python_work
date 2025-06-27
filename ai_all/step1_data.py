import pandas as pd

# 模擬的資料
data = {
    "avg_speed": [35, 10, 60, 20, 55, 12, 50, 18, 65, 13],  # 速度
    "acc_var": [2.0, 4.5, 1.2, 4.9, 1.1, 4.8, 1.3, 5.1, 1.0, 4.6],  # 加速度變異
    "label": ["car", "scooter", "car", "scooter", "car", "scooter", "car", "scooter", "car", "scooter"]
}

df = pd.DataFrame(data)

# 儲存成 CSV
df.to_csv("vehicle_data.csv", index=False)

print("✅ 已建立並儲存模擬資料 vehicle_data.csv")
