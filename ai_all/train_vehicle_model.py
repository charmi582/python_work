import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 建立模擬資料（每類 100 筆）
def generate_samples(label, speed_range, acc_range, n=100):
    return pd.DataFrame({
        "avg_speed": np.random.uniform(*speed_range, n).round(2),
        "acc_var": np.random.uniform(*acc_range, n).round(2),
        "label": [label] * n
    })

walk_data = generate_samples("walk", (0.5, 5.5), (1.5, 3.5))
bike_data = generate_samples("bike", (10, 20), (2.5, 4.5))
scooter_data = generate_samples("scooter", (25, 45), (3.5, 5.5))
car_data = generate_samples("car", (30, 70), (0.5, 2.5))

df = pd.concat([walk_data, bike_data, scooter_data, car_data]).sample(frac=1, random_state=42).reset_index(drop=True)

# 建模流程
X = df[["avg_speed", "acc_var"]].values
y = df["label"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 模型架構
model = Sequential([
    Dense(32, input_shape=(2,), activation='relu'),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型訓練
model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

# 評估準確率
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ 測試準確率：{accuracy * 100:.2f}%")

# 儲存模型與標籤
model.save("vehicle_classifier_model_v2.h5")
np.save("vehicle_label_classes_v2.npy", label_encoder.classes_)
print("✅ 模型與標籤已儲存")
