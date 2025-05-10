from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[8, 1], [7, 2], [6.5, 3], [5.5, 4.5], [5, 5.5], [4.5, 6], [7, 3.5], [6, 4]]
y=[0, 0, 0, 1, 1, 1, 0, 1]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:", acc)

now_data=[[7.5, 3]]
print("當我使用3小時的手機，睡7.5個小時，這樣會睡飽嗎?", model.predict(now_data)[0])