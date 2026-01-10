import torch
import torch.nn as nn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp import MLP
from optimizers import Adam


# -------------------------------
# Load dataset
# -------------------------------
data = load_diabetes()
X = data.data
y = data.target.reshape(-1, 1)

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# -------------------------------
# Custom MLP
# -------------------------------
model = MLP(
    input_dim=X_train.shape[1],
    hidden_dims=[32, 32],
    output_dim=1,
    activation=nn.ReLU
)

optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 300

for epoch in range(epochs):
    preds = model(X_train)
    loss = torch.mean((preds - y_train) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"[Custom MLP] Epoch {epoch} | Loss: {loss.item():.4f}")

with torch.no_grad():
    test_preds = model(X_test)
    test_loss = torch.mean((test_preds - y_test) ** 2)

print("\nCustom MLP Test Loss:", test_loss.item())


# -------------------------------
# PyTorch Baseline
# -------------------------------
torch_model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)

for epoch in range(epochs):
    torch_optimizer.zero_grad()
    preds = torch_model(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    torch_optimizer.step()

    if epoch % 50 == 0:
        print(f"[Torch MLP] Epoch {epoch} | Loss: {loss.item():.4f}")

with torch.no_grad():
    torch_test_preds = torch_model(X_test)
    torch_test_loss = criterion(torch_test_preds, y_test)

print("Torch MLP Test Loss:", torch_test_loss.item())
