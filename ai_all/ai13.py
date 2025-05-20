from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

x=np.array([[5.0, 4], [6.0, 5], [7.5, 8], [6.5, 6], [8.0, 9], [7.0, 7], [5.5, 3], [6.2, 4]])
y=np.array([1, 1, 0, 1, 0, 0, 1, 1])

model=LogisticRegression()
model.fit(x, y)

for label in np.unique(y):
    plt.scatter(x[y==label][:, 0], x[y==label][:, 1],
    label=f"是否毀喝咖啡={label}", s=100)
    
x_min, x_max=x[:, 0].min()-0.5, x[:, 0].max()+0.5
y_min, y_max=x[:, 1].min()-0.5, x[:, 1].max()+0.5

xx, yy=np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300))

grid=np.c_[xx.ravel(), yy.ravel()]

z=model.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.2, cmap=plt.cm.RdYlBu)
plt.grid(True)
plt.show()