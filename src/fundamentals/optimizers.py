# optimizers.py
# Optimizers for our custom neural network engine
# They operate directly on parameter.data using parameter.grad

import math
import torch

class Optimizer:
    """
    Base optimizer class.
    Stores parameters and defines the interface.
    """

    def __init__(self, params):
        self.params = list(params)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    Update rule:
        param = param - lr * grad
    """

    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad


class SGDMomentum(Optimizer):
    """
    SGD with Momentum

    velocity = beta * velocity + grad
    param = param - lr * velocity
    """

    def __init__(self, params, lr=0.01, beta=0.9):
        super().__init__(params)
        self.lr = lr
        self.beta = beta
        self.velocity = {}

        for p in self.params:
            self.velocity[p] = 0.0

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            v = self.velocity[p]
            v = self.beta * v + p.grad
            self.velocity[p] = v

            p.data -= self.lr * v


class Adam(Optimizer):
    """
    Adam Optimizer

    Maintains:
        m -> first moment (mean of gradients)
        v -> second moment (mean of squared gradients)

    Bias-corrected updates are applied.
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

        for p in self.params:
            self.m[p] = 0.0
            self.v[p] = 0.0

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad
            # First moment
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g

            # Second moment
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (g ** 2)

            #Bias correction
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
