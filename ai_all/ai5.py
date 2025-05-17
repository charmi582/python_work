from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[1, 8], [1, 7], [1, 4], [0, 8], [0, 3], [0, 1], [1, 2], [1, 6]]
y=[0, 0, 1, 1, 1, 1, 1, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)
model=LogisticRegression
model.fit(x_train, y_train)

pred=model.predict(x_test)
acc=accuracy_score(x_test, pred)
print("準確率:", acc)

now_data=[[0, 5]]
print("今天下雨、煮飯意願5分，會叫外送嗎？預測為：", model.predict(now_data)[0])
        