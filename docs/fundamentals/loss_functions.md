# Loss Functions as Modules

This document explains how loss functions work internally and maps
directly to the implementation in `loss_functions.py`.

Loss functions are the **bridge between predictions and learning**.

---

## What Is a Loss Function?

A loss function:
- Measures prediction error
- Outputs a scalar tensor
- Serves as the starting point for backpropagation

Calling:

loss.backward()

propagates gradients from the loss back through the entire network.

---

## Why Loss Functions Are Modules

Treating loss functions as callable objects:
- Matches PyTorch’s design (`nn.MSELoss`, `nn.CrossEntropyLoss`)
- Makes training loops clean and extensible
- Keeps math isolated from training logic

---

## Mean Squared Error (MSE)

### Use Case
Regression problems.

### Formula
loss = mean((prediction − target)²)

### Behavior
- Penalizes large errors heavily
- Smooth gradient
- Sensitive to outliers

### Code Mapping

```python
diff = prediction - target
loss = (diff ** 2).mean()




Binary Cross Entropy (BCE)
Use Case

Binary classification (0 or 1).

Assumptions

Predictions are probabilities (sigmoid applied)

Targets are 0 or 1

Formula

loss = −[y·log(p) + (1−y)·log(1−p)]

Numerical Stability

Predictions are clamped to avoid log(0).

Cross Entropy Loss (Multi-Class)
Use Case

Multi-class classification.

Inputs

Logits (raw model outputs)

Target class indices

Key Insight

Softmax and log are combined for stability.

Steps

Apply log-softmax

Select correct class probabilities

Compute negative log-likelihood

Code Mapping
log_probs = log_softmax(logits)
loss = -log_probs[correct_class].mean()

How Loss Fits into Training

Typical sequence:

prediction = model(x)
loss = loss_fn(prediction, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()


What changes:

loss.backward() → computes gradients

optimizer.step() → updates parameters

zero_grad() → prevents accumulation

Why This Matters

Understanding loss functions:

Explains why gradients behave the way they do

Prevents misuse (e.g. softmax + cross entropy twice)

Prepares you for custom research losses

Summary
Loss	Task
MSE	Regression
BCE	Binary classification
Cross Entropy	Multi-class classification