import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

x, y=make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

x_train=torch.tensor(x_train, dtype=torch.float32)
y_train=torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test=torch.tensor(x_test, dtype=torch.float32)
y_test=torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class DNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net=nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
model=DNN()

loss_fn=nn.BCELoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    y_pred=model(x_train)
    loss=loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%10==0:
        print(f"epoch{epoch+1}: loss = {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        y_pred_test=model(x_test)
        y_pred_label=(y_pred_test>0.5).float()
        acc=accuracy_score(y_test.numpy(), y_pred_label.numpy())
        print(f"準確率{acc:.4f}")