

# 2️⃣ `training_loop.py`  


import torch
import torch.nn as nn


# --------------------------------------------------
# Simple linear model
# --------------------------------------------------
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# --------------------------------------------------
# Synthetic dataset
# --------------------------------------------------
def generate_data():
    x = torch.linspace(0, 10, steps=100).unsqueeze(1)
    y = 2 * x + 1
    return x, y


# --------------------------------------------------
# Training loop
# --------------------------------------------------
def train(model, inputs, targets, loss_fn, optimizer, epochs=100):
    for epoch in range(epochs):

        # 1. Clear accumulated gradients
        optimizer.zero_grad()

        # 2. Forward pass
        predictions = model(inputs)

        # 3. Compute loss
        loss = loss_fn(predictions, targets)

        # 4. Backward pass (compute gradients)
        loss.backward()

        # 5. Update parameters
        optimizer.step()

        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")


# --------------------------------------------------
# Main execution
# --------------------------------------------------
def main():
    model = SimpleLinearModel()
    x, y = generate_data()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train(
        model=model,
        inputs=x,
        targets=y,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=100
    )

    print("\nLearned parameters:")
    print("Weight:", model.linear.weight.item())
    print("Bias:", model.linear.bias.item())


if __name__ == "__main__":
    main()
