from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

x=[[7.0, 8], [6.5, 7.5], [8.0, 6.0], [7.2, 5.5], [6.8, 6.5], [7.5, 5.0], [7.0, 6.0], [8.3, 6.0]]
y=[1, 1, 0, 0, 1, 0, 1, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_pred, y_test)
print("準確率:", acc)

now_data=[[7.2, 6.5]]
print("起床 7.2 點、睡 6.5 小時，今天會準時嗎？預測為：", model.predict(now_data)[0])