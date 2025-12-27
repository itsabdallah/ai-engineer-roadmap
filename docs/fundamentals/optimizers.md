Optimizers (Implementation-Focused)

An optimizer updates model parameters using gradients computed by autograd.
It does not compute gradients — it only consumes param.grad and updates param.data.

The standard training step looks like:

loss.backward()
optimizer.step()
optimizer.zero_grad()


Everything below explains what happens inside optimizer.step().

Parameters and Gradients

Each trainable parameter is a torch.Tensor with:

param.data → parameter values

param.grad → gradient of the loss w.r.t. the parameter

Optimizers iterate over parameters and apply update rules using param.grad.

1. Stochastic Gradient Descent (SGD)
Update Rule
param = param − lr × grad

Code Mapping
param.data -= lr * param.grad

Notes

Single global learning rate

Simple and predictable

Sensitive to learning rate choice

2. SGD with Momentum

Momentum introduces a velocity term to accumulate past gradients and smooth updates.

State per Parameter

velocity

Update Rule
velocity = β · velocity + grad
param = param − lr · velocity

Code Mapping
velocity = beta * velocity + param.grad
param.data -= lr * velocity

Effect

Reduces oscillations

Accelerates convergence in consistent directions

3. Adam Optimizer

Adam combines:

Momentum (first moment)

Adaptive learning rates (second moment)

Each parameter maintains two running estimates.

State per Parameter

m → first moment (mean of gradients)

v → second moment (mean of squared gradients)

t → time step

Update Rules

First moment:

m = β₁ · m + (1 − β₁) · grad


Second moment:

v = β₂ · v + (1 − β₂) · grad²


Bias correction:

m̂ = m / (1 − β₁ᵗ)
v̂ = v / (1 − β₂ᵗ)


Parameter update:

param = param − lr · m̂ / (sqrt(v̂) + ε)

Why Adam Is Widely Used

Stable updates

Adaptive per-parameter step sizes

Works well with sparse or noisy gradients

Less sensitive to learning rate tuning

Optimizer State

Optimizers maintain internal state, separate from the model:

momentum buffers

running averages

step counters

This is why full checkpoints save both:

model.state_dict()
optimizer.state_dict()

Gradient Reset (zero_grad)

Gradients accumulate by default in PyTorch.

Correct usage:

optimizer.zero_grad()
loss.backward()
optimizer.step()

Summary
Component	Role
param.data	Parameter values
param.grad	Gradient from autograd
Optimizer	Updates parameters
Momentum	Smooths updates
Adam	Adaptive + momentum
zero_grad()	Prevents accumulation