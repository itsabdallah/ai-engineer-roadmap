# Multi-Layer Neural Networks (From Scratch)

This document explains how a **multi-layer neural network** works by directly mapping concepts to the implementation in `multi_layer.py`.

The goal is not to use high-level PyTorch abstractions, but to understand **what actually happens inside a deep network**.

---

## What Is a Multi-Layer Network?

A multi-layer neural network (also called a **deep neural network**) is a stack of layers where each layer applies:

Linear transformation → Non-linear activation

Mathematically, each layer computes:

output = activation(input × weights + bias)

Stacking multiple layers allows the network to learn **non-linear and hierarchical representations** that a single layer cannot capture.

---

## Why Multiple Layers Matter

A single linear layer can only learn linear relationships.

By stacking layers and inserting **non-linear activations** (e.g. ReLU), the network can:

- Model complex patterns
- Learn hierarchical features
- Approximate arbitrary functions

Depth increases **expressive power**, not just parameter count.

---

## Linear Layer (Fully Connected Layer)

Each linear layer performs:

output = x @ W + b

Where:
- `x` is the input tensor
- `W` is the weight matrix
- `b` is the bias vector

In `multi_layer.py`, this is implemented manually:

- `W` and `b` are `torch.Tensor`s with `requires_grad=True`
- Autograd tracks operations on them automatically

Each layer exposes a `parameters()` method so optimizers can update them.

---

## Activation Functions (ReLU)

After each linear transformation, a **non-linear activation** is applied.

ReLU (Rectified Linear Unit):

ReLU(x) = max(0, x)

Why ReLU:
- Prevents linear collapse
- Efficient gradient flow
- Widely used in practice

In the implementation, ReLU is applied using:

```python
torch.nn.functional.relu(x)


---

## Network Architecture (MLP) 
The Multi-Layer Perceptron (MLP) in multi_layer.py uses:

-Input layer
-Two hidden layers
-Output layer

Structure:

Input → Linear → ReLU → Linear → ReLU → Linear → Output

Hidden layers learn intermediate representations.
The output layer produces final predictions.

No activation is applied to the output for regression tasks.

---

## Forward Pass
The forward pass propagates input through the network:

1. Input enters first linear layer
2. Activation applied
3. Passed to next layer
4. Repeated until output

Each operation builds a computation graph tracked by autograd.

---

## Loss Computation
A loss function measures prediction error.

In this implementation:

Mean Squared Error (MSE):

loss = mean((prediction − target)²)

The loss is a scalar tensor connected to all parameters via the computation graph.

---

## Backpropagation
Calling:

loss.backward()


Triggers automatic differentiation.

What happens:

-Gradients are computed for every parameter
-Gradients are stored in param.grad
-No parameters are updated yet

Autograd applies the chain rule across all layers.

## Parameter Updates (Manual SGD)
After gradients are computed, parameters are updated manually:

param = param − learning_rate × grad

This step:

Uses .data implicitly via torch.no_grad()

Modifies parameters in place

Does not build a new computation graph

Gradients are reset afterward using:

param.grad.zero_()


---

## What Changes After Each Step?
After loss.backward():

-param.grad contains gradients
-Parameters remain unchanged

After parameter update:

-Parameters change
-Model behavior changes
-Loss should decrease over time

---

## Why This Matters?
This file demonstrates:

-How depth works
-How layers interact
-How gradients flow across layers
-How training actually updates a model

This is the conceptual foundation of all modern neural networks, including CNNs, RNNs, and Transformers.

---

## Relation to PyTorch High-Level APIs
This manual implementation mirrors what PyTorch does internally with:

-nn.Linear
-nn.Module
-optim.Optimizer

Understanding this level allows you to:

-Debug training issues
-Modify architectures confidently
-Read research code with clarity

---

##Summary
Component               	Role
Linear Layer	            Learns affine transformation
ReLU	                    Introduces non-linearity
Multiple Layers	            Increase expressive power
Loss	                    Measures error
Backward	                Computes gradients
Update Step	                Improves parameters