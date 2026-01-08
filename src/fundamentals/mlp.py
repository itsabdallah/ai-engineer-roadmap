"""
Multi-Layer Perceptron (MLP) implementation
Built from scratch using custom engine components.
"""

import torch


class Linear:
    """
    Fully connected linear layer: y = xW + b
    """

    def __init__(self, in_features, out_features):
        # Create leaf tensors FIRST
        W = torch.randn(in_features, out_features) * 0.01
        W.requires_grad_()
        self.W = W

        b = torch.zeros(out_features)
        b.requires_grad_()
        self.b = b

    def __call__(self, x):
        """
        Forward pass
        """
        return x @ self.W + self.b

    def parameters(self):
        """
        Return parameters for optimizer
        """
        return [self.W, self.b]


class MLP:
    """
    Multi-Layer Perceptron composed of Linear layers and activations.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation):
        self.layers = []

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                self.layers.append(activation())

    def __call__(self, x):
        """
        Forward pass through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Collect parameters from all layers
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params
