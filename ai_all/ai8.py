from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[1.0, 6.5], [2.5, 7.0], [3.5, 8.0], [4.0, 8.3], [1.2, 6.8], [3.0, 7.9], [2.0, 7.2], [4.5, 8.5]]
y=[1, 1, 0, 0, 1, 0, 1, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:", acc)

now_data=[[3.2, 7.9]]
result=model.predict(now_data)[0]
if result==0:
    print("不會來上課")
elif result==1:
    print("會來上課")