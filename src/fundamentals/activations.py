"""
Activation functions implemented as modules.
These operate on Tensors and integrate with the autograd engine.
"""

import torch


class ReLU:
    def __call__(self, x):
        """
        ReLU(x) = max(0, x)
        """
        return torch.maximum(x, torch.zeros_like(x))


class Sigmoid:
    def __call__(self, x):
        """
        Sigmoid(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + torch.exp(-x))


class Tanh:
    def __call__(self, x):
        """
        Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        """
        return torch.tanh(x)
