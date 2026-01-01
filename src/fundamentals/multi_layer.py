"""
Multi-Layer Neural Network (From Scratch Conceptually)

This file demonstrates how a multi-layer (deep) neural network is built
using linear layers and non-linear activation functions.

Key ideas:
- Stacking layers increases representational power
- Each layer applies: Linear -> Activation
- Autograd handles gradient computation automatically
"""

import torch
import torch.nn.functional as F


class LinearLayer:
    """
    A fully connected (dense) layer implemented manually.

    Performs:
        output = input @ W + b
    """

    def __init__(self, in_features, out_features):
        # Weight matrix: (in_features, out_features)
        self.W = torch.randn(in_features, out_features, requires_grad=True)

        # Bias vector: (out_features,)
        self.b = torch.zeros(out_features, requires_grad=True)

    def forward(self, x):
        """
        Forward pass through the linear layer
        """
        return x @ self.W + self.b

    def parameters(self):
        """
        Returns parameters so an optimizer can update them
        """
        return [self.W, self.b]


class MLP:
    """
    Multi-Layer Perceptron (MLP)

    Architecture:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = LinearLayer(input_dim, hidden_dim)
        self.layer2 = LinearLayer(hidden_dim, hidden_dim)
        self.layer3 = LinearLayer(hidden_dim, output_dim)

    def forward(self, x):
        # First hidden layer
        x = self.layer1.forward(x)
        x = F.relu(x)

        # Second hidden layer
        x = self.layer2.forward(x)
        x = F.relu(x)

        # Output layer (no activation for regression)
        x = self.layer3.forward(x)
        return x

    def parameters(self):
        """
        Collect all parameters from all layers
        """
        params = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            params.extend(layer.parameters())
        return params


def main():
    torch.manual_seed(42)

    # Dummy input batch: (batch_size, input_dim)
    x = torch.randn(5, 3)

    # Dummy target: (batch_size, output_dim)
    y = torch.randn(5, 1)

    # Create model
    model = MLP(input_dim=3, hidden_dim=8, output_dim=1)

    # Forward pass
    predictions = model.forward(x)

    # Mean Squared Error loss
    loss = torch.mean((predictions - y) ** 2)

    print("Loss before backward:", loss.item())

    # Backpropagation
    loss.backward()

    # Inspect gradients
    print("\nGradient shapes:")
    for param in model.parameters():
        print(param.shape, "-> grad shape:", param.grad.shape)

    # Manual SGD step (no optimizer abstraction yet)
    learning_rate = 0.01
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            param.grad.zero_()

    # Forward pass after update
    new_predictions = model.forward(x)
    new_loss = torch.mean((new_predictions - y) ** 2)

    print("\nLoss after one update step:", new_loss.item())


if __name__ == "__main__":
    main()
