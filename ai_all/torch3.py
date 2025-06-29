import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

x, y=make_classification(n_samples=1000, n_classes=3, n_features=6, n_informative=6, n_redundant=0,
                         random_state=42)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, stratify=y)

x_train=torch.tensor(x_train, dtype=torch.float32)
y_train=torch.tensor(y_train, dtype=torch.long)
x_test=torch.tensor(x_test, dtype=torch.float32)
y_test=torch.tensor(y_test, dtype=torch.long)

class DNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net=nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)
model=DNN()

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    y_pred=model(x_train)
    loss=loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    model.eval()
    with torch.no_grad():
        y_pred_test=model(x_test)
        y_pred_test=torch.argmax(y_pred_test, dim=1)
        acc=accuracy_score(y_test.numpy(), y_pred_test.numpy())
        print(f"準確率:{acc:.4f}")


    

