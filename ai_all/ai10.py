from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

x= np.array([[2, 1.0], [4, 1.5], [5, 2.5], [6, 3.5], [3, 1.2], [7, 4.5], [5, 2.8], [4, 1.3]])
y=np.array([0, 0, 1, 1, 0, 1, 1, 0])

model= LogisticRegression()
model.fit(x, y)

for label in np.unique(y):
    plt.scatter(x[y==label][:, 0], x[y==label][:, 1],
                label=f"是否熬夜={label}", s=100)
    
    
x_min, x_max = x[:, 0].min()-0.5, x[:, 0].max()+0.5
y_min, y_max = x[:, 1].min()-0.5, x[:, 1].max()+0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlBu)

label=f"是否熬夜={label}"    
plt.xlabel("壓力程度")       
plt.ylabel("手機使用時間")    
plt.title("AI 分類視覺化：是否熬夜")  
plt.show()