from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

x=[[8, 1], [7, 1.5], [6, 2], [5, 2.5], [4, 3], [3, 3.5], [2, 4]]
y=[1, 1, 1, 0, 0, 0, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model=LogisticRegression
model.fit=(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print=("精確值:", acc)

new_data=[[6.25, 2]]
print("使用2.5小時的手機稅6小時，會有精神嗎?", model.predict(new_data)[0])