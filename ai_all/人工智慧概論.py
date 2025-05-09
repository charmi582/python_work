import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 建立資料（餐廳範例）
data = {
    'Alternate'    : ['Yes','Yes','No', 'Yes','Yes', 'No', 'No', 'No', 'No', 'Yes','No', 'Yes'],
    'Bar'          : ['No','No','Yes','No','Yes','Yes','No','No','No','Yes','Yes','Yes'],
    'Fri/Sat'      : ['No','Yes','No', 'Yes','No', 'No', 'Yes','Yes','No', 'Yes','No', 'Yes'],
    'Hungry'       : ['Yes','No','Yes','No','Yes','No','Yes','No','Yes','No','Yes','Yes'],
    'Patrons'      : ['Some','Full','Some','Full','Full','Some','None','Some','Full','Full','None','Full'],
    'Price'        : ['$$$','$','$$$','$$','$','$$$','$','$','$$$','$$','$','$'],
    'Raining'      : ['No','No','Yes','No','Yes','Yes','Yes','No','No','Yes','No','No'],
    'Reservation'  : ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','No','Yes','No'],
    'Type'         : ['French','Thai','Burger','Thai','French','Italian','Burger','Thai','Burger','Italian','Thai','Burger'],
    'WaitEstimate' : ['0-10','30-60','0-10','10-30','>60','0-10','0-10','0-10','>60','10-30','0-10','30-60'],
    'WillWait'     : ['Yes','No','Yes','No','Yes','No','No','No','No','Yes','No','Yes']
}
df = pd.DataFrame(data)

# 2. 資料前處理：將類別型變數編碼為數值
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # 儲存編碼器以供轉換或解碼

# 3. 特徵與標籤分開
X = df.drop('WillWait', axis=1)
y = df['WillWait']

# 4. 建立並訓練決策樹分類器
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
clf.fit(X, y)

# 5. 模型預測與準確率
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print("訓練準確率：", accuracy)

# 6. 視覺化決策樹
plt.figure(figsize=(20,10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=label_encoders['WillWait'].classes_,
    filled=True,
    rounded=True
)
plt.title("餐廳等待決策樹")
plt.show()
