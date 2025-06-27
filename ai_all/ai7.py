from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[30, 0], [120, 1], [15, 0], [200, 1], [60, 1], [40, 0], [80, 1], [25, 0]]
y=[0, 1, 0, 1, 1, 0, 1, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

model= LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_pred, y_test)
print("準確率:", acc)

now_data=[[90, 1]]
result=model.predict(now_data)[0]

if result==1:
    print("會留言")
elif result==0:
    print("不會留言")