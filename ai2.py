# 第一次完整的 AI 模型（交通方式分類器）
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假設我們的資料長這樣：
# 特徵：平均速度(km/h)、加速度變化量(m/s^2)
# 標籤：0 = scooter（機車），1 = car（汽車）
X = [
    [30, 4],  # scooter
    [35, 5],
    [25, 6],
    [80, 2],  # car
    [90, 1],
    [85, 1.5],
    [40, 3],  # scooter
    [45, 4],
    [70, 2],  # car
    [33, 5]   # scooter
]
y = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立邏輯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("準確率：", acc)
