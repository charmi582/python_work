import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

np.random.seed(10)  # 指定亂數種子
# 載入鳶尾花資料集
iris = load_iris()
X = iris.data[:, :2]     # 使用花萼長度與寬度
y = iris.target          # 0: Setosa, 1: Versicolor, 2: Virginica

# 標準化特徵
X -= X.mean(axis=0)
X /= X.std(axis=0)

# One-hot 編碼標籤（用於多分類 softmax）
y_cat = to_categorical(y, num_classes=3)

# 分割訓練集與測試集
X_train, y_train = X[:120], y_cat[:120]   # 前120筆為訓練
X_test, y_test = X[120:], y_cat[120:]     # 後30筆為測試

# 定義模型：模擬感知器（單層 softmax）
model = Sequential()
model.add(Input(shape=(2,)))              # 2個輸入特徵
model.add(Dense(3, activation="softmax")) # 3類輸出

# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# 訓練模型
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)

# 評估準確率
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
print("--------------------------")

# 測試資料的預測
y_pred = model.predict(X_test, batch_size=5, verbose=0)
print("第一筆預測 softmax 結果：", y_pred[0])
print("預測類別為：", np.argmax(y_pred[0]))
