# Multi-Layer Perceptron (MLP) — From Scratch Implementation

This document explains the full implementation of a Multi-Layer Perceptron (MLP)
built from first principles using PyTorch tensors and autograd, without relying
on `torch.nn.Module`.

The goal is to understand how neural networks are constructed internally:
layers, parameters, forward passes, and gradient flow.

---


## 1. What Is an MLP?


A Multi-Layer Perceptron (MLP) is a feedforward neural network composed of:

- Linear (fully connected) layers
- Non-linear activation functions
- A final output layer

Each layer performs the following computation:
x → xW + b → activation(xW + b)


By stacking multiple layers, the network can learn non-linear functions.

---

## 2. Design Philosophy

This MLP implementation is intentionally minimal and explicit:

- No inheritance from `torch.nn.Module`
- No automatic parameter registration
- No built-in optimizers
- Explicit parameter handling

This exposes the mechanics that are normally hidden by high-level APIs.

---

## 3. Linear Layer

### Purpose

The `Linear` layer represents a fully connected transformation:
y = x @ W + b


Where:
- `W` is the weight matrix
- `b` is the bias vector

---





### Parameter Initialization

Weights and biases are created as **leaf tensors**:

Parameters are created as **leaf tensors** to ensure gradients are populated.


    W = torch.randn(in_features, out_features) * 0.01

    W.requires_grad_()

    b = torch.zeros(out_features)

    b.requires_grad_()

### Key Points

- Layers are applied sequentially
- Each layer receives the output of the previous layer
- PyTorch autograd automatically tracks all operations
- No manual gradient wiring is required

Each forward pass builds a dynamic computation graph that will later be
used during backpropagation.

---

## 4. MLP Construction

The MLP class composes multiple layers into a single model.

Layer Dimensions:
--
    dims = [input_dim] + hidden_dims + [output_dim]


This allows arbitrary depth:

1. Input layer

2. One or more hidden layers

3. Output layer


Activation Placement
---
    if i < len(dims) - 2:
    self.layers.append(activation())


1. Activations are applied between linear layers

2. The output layer has no activation

3. The activation function is passed as a class, not a function

This design enables easy swapping of activation functions.

---
## 5. Forward Pass Through the Network
    def __call__(self, x):
    for layer in self.layers:
        x = layer(x)
    return x


1. Data flows sequentially through all layers

2. Each layer transforms the output of the previous one

3. Autograd builds the computation graph automatically

---
## 6. Parameter Collection

To train the model, we must expose **all trainable parameters** (weights and
biases) to the optimizer.

This is handled by iterating through all layers and collecting parameters
only from layers that define them.


    def parameters(self):
     params = []
    for layer in self.layers:
        if hasattr(layer, "parameters"):
            params.extend(layer.parameters())
    return params



### Why This Works

- Linear layers define a `parameters()` method
- Activation layers do not have trainable parameters
- The model simply skips layers without parameters
- The result is a flat list of tensors

This design mirrors how deep learning frameworks internally manage
parameters while remaining fully explicit and transparent.

---

## 7. Gradient Flow and Autograd

Gradients are computed by calling:


    loss.backward()


When this is executed, PyTorch:

1. Traverses the computation graph backward

2. Applies the chain rule

3. Accumulates gradients into .grad fields

Important Rule: Leaf Tensors Only
---

Only leaf tensors receive gradients.

A tensor is a leaf tensor if:

1. It was created directly by the user

2. It has requires_grad=True

3. It is not the result of an operation

This is why parameters must be created explicitly and not via expressions.

---
## 8. Training Step (Conceptual)
A single training iteration follows this standard pattern:

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

Step-by-Step Explanation
---

1. zero_grad() clears gradients from the previous step

2. backward() computes gradients via backpropagation

3. step() updates parameters using those gradients

In this project, optimizers are implemented manually to make updates explicit.

---
## 9. Why This Is Truly "From Scratch"
Although PyTorch tensors are used, this implementation avoids:

1. torch.nn.Module

2. torch.optim

3. Automatic parameter registration

Everything is done manually:
--
1. Layers are explicitly composed

2. Parameters are explicitly collected

3. Updates are explicitly applied

This exposes the internal mechanics of deep learning systems.

---

## 10. Summary
1. An MLP is a stack of linear layers and activations

2. Forward passes build computation graphs

3. Backward passes compute gradients automatically

4. Only leaf tensors receive gradients

5. Parameter updates are fully transparent

This implementation forms the foundation for:
--
1. Custom optimizers

2. Custom loss functions

3. Training loops

4. More advanced architectures