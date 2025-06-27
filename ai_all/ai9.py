from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x=[[2, 0], [10, 1], [5, 0], [12, 1], [3, 0], [9, 1], [6, 0], [11, 1]]
y=[1, 0, 1, 0, 1, 0, 1, 0]

x_train, x_test, y_train, y_test=train_test_split(x, y, train_size=0.3,random_state=42, stratify=y)

model=LogisticRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("準確率:", acc)

now_data=[[4, 0]]
result=model.predict(now_data)[0]
print("會繼續看"if result==1 else "不會繼續看")