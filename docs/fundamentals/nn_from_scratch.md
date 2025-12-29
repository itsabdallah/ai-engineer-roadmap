Neural Network From Scratch

This document explains a neural network implemented from first principles, without using torch.nn or torch.optim. The objective is to understand the mechanics that PyTorch abstracts away in higher-level APIs.

Only low-level PyTorch components are used:

torch.Tensor

Autograd

Manual parameter updates

Model Definition

We implement a simple linear regression model:

y = w · x + b

Where:

w is a learnable weight parameter

b is a learnable bias parameter

Both parameters are tensors with requires_grad=True, allowing PyTorch’s autograd engine to track operations and compute gradients automatically.

Forward Pass

The forward pass computes predictions using basic tensor operations:

y_pred = w * x + b


This builds a computation graph connecting inputs, parameters, and outputs.
No gradients are computed during the forward pass.

Loss Function

We use Mean Squared Error (MSE) to measure prediction error:

loss = ((y_pred - y_true) ** 2).mean()


The loss is a scalar tensor representing how far predictions are from the target values.

Backward Pass (Autograd)

Calling:

loss.backward()


Triggers automatic differentiation.

Autograd traverses the computation graph in reverse and computes gradients:

w.grad → ∂loss / ∂w

b.grad → ∂loss / ∂b

At this stage:

Parameters have not changed

Only gradients have been computed

Parameter Update (Manual Optimization)

We manually apply Stochastic Gradient Descent (SGD):

with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad


Important details:

torch.no_grad() prevents tracking these updates

Gradients are read but not modified

Parameters are updated in-place

This step is functionally equivalent to optimizer.step().

Gradient Reset

PyTorch accumulates gradients by default.

After each update, gradients must be cleared:

w.grad.zero_()
b.grad.zero_()


This prevents gradients from accumulating across iterations and is equivalent to calling optimizer.zero_grad().

Training Loop Structure

Each training iteration follows this sequence:

Forward pass

Loss computation

Backward pass (loss.backward())

Parameter update

Gradient reset

This structure is the foundation of all neural network training workflows.

Mapping to PyTorch Abstractions
Manual Implementation	PyTorch Abstraction
Tensors with requires_grad	nn.Parameter
Loss computation	nn.MSELoss
Manual updates	optimizer.step()
Gradient reset	optimizer.zero_grad()
Key Takeaways

Autograd computes gradients, not parameter updates

Optimizers apply update rules using gradients

High-level APIs automate this exact workflow

Understanding this process enables better debugging, customization, and research-level control