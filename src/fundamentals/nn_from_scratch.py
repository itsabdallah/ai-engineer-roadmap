import torch

# --------------------------------------------------
# Neural Network From Scratch (No torch.nn, No torch.optim)
# --------------------------------------------------
# Goal:
# - Build a simple neural network manually
# - Use tensors, autograd, and manual optimization
# - Understand exactly what PyTorch automates later
# --------------------------------------------------


# -------------------------
# 1. Create Training Data
# -------------------------
# Simple linear relationship: y = 2x + 1
# This keeps the focus on mechanics, not data complexity

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])


# -------------------------
# 2. Initialize Parameters
# -------------------------
# We manually define weights and bias
# requires_grad=True tells autograd to track these tensors

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


# -------------------------
# 3. Define Forward Pass
# -------------------------
# This is the model: y_pred = w * x + b
# No torch.nn.Module — just raw tensor ops

def forward(x):
    return w * x + b


# -------------------------
# 4. Define Loss Function
# -------------------------
# Mean Squared Error (MSE)
# loss = average((prediction - target)^2)

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


# -------------------------
# 5. Training Hyperparameters
# -------------------------
learning_rate = 0.01
epochs = 100


# -------------------------
# 6. Training Loop
# -------------------------
for epoch in range(epochs):

    # ---- Forward pass ----
    y_pred = forward(x)

    # ---- Compute loss ----
    loss = mse_loss(y_pred, y)

    # ---- Backward pass ----
    # Computes gradients:
    # w.grad and b.grad are populated here
    loss.backward()

    # ---- Parameter update (manual SGD) ----
    # torch.no_grad() because parameter updates
    # should NOT be tracked by autograd
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # ---- Clear gradients ----
    # Gradients accumulate by default
    w.grad.zero_()
    b.grad.zero_()

    # ---- Logging ----
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"w: {w.item():.4f} | "
            f"b: {b.item():.4f}"
        )


# -------------------------
# 7. Final Model Check
# -------------------------
print("\nFinal parameters:")
print(f"w ≈ {w.item():.4f}")
print(f"b ≈ {b.item():.4f}")

print("\nPrediction for x = 5:")
test_x = torch.tensor([5.0])
print(forward(test_x).item())
