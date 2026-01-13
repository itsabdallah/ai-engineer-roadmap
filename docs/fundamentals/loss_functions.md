# Loss Functions as Modules

This document explains loss functions from an implementation-focused perspective.
It maps directly to `loss_functions.py` and follows the same structure and logic.

Loss functions are the final component in the forward pass and the starting point
of backpropagation.

---

## What Is a Loss Function?

A loss function measures how incorrect a model’s predictions are.

It:
- Takes model predictions and ground-truth targets
- Produces a single scalar value
- Initiates gradient computation via `loss.backward()`

The loss itself does not update parameters.
It only provides the signal from which gradients are computed.

---

## Why Loss Functions Are Implemented as Modules

Implementing loss functions as callable classes:
- Matches PyTorch’s design (`nn.MSELoss`, `nn.CrossEntropyLoss`)
- Keeps training loops clean
- Separates math from optimization logic
- Makes it easy to swap losses without changing training code

Each loss object behaves like a function:

loss = loss_fn(prediction, target)


---

## Mean Squared Error (MSE)

### Use Case
Regression problems where targets are continuous values.

### Definition
Mean Squared Error computes the average squared difference between predictions and targets.

### Formula
loss = mean((prediction − target)²)

### Behavior
- Penalizes large errors more than small ones
- Produces smooth gradients
- Sensitive to outliers

### Code Mapping

    diff = prediction - target
    squared_diff = diff ** 2
    loss = squared_diff.mean()


 
---

## Binary Cross Entropy (BCE)

### Use Case
Binary classification tasks (labels are 0 or 1).

### Assumptions
- Model output is a probability between 0 and 1
- Sigmoid has already been applied

### Definition
Binary Cross Entropy measures the distance between predicted probabilities and true binary labels.

### Formula
    loss = −[y · log(p) + (1 − y) · log(1 − p)]

### Numerical Stability
Predictions are clamped to avoid log(0), which would produce NaNs.

### Code Mapping

    prediction = clamp(prediction)
    loss = -(target * log(prediction)
    + (1 - target) * log(1 - prediction))
    loss = loss.mean()



---

## Cross Entropy Loss (Multi-Class)

### Use Case
Multi-class classification problems.

### Inputs
- Logits (raw model outputs, no softmax applied)
- Target class indices (not one-hot encoded)

### Key Insight
Softmax and logarithm are combined using log-softmax for numerical stability.

### Steps
1. Apply log-softmax to logits
2. Select log-probabilities of correct classes
3. Compute negative log-likelihood

### Formula
    loss = −mean(log(p_correct_class))

### Code Mapping

    log_probs = log_softmax(logits)
    correct_log_probs = log_probs[batch_indices, target]
    loss = -correct_log_probs.mean()



---

## Loss Functions in the Training Loop

Typical training sequence:
  
    prediction = model(x)

    loss = loss_fn(prediction, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()




What happens:
- `loss.backward()` computes gradients
- `optimizer.step()` updates parameters
- `optimizer.zero_grad()` clears accumulated gradients

---

## Important Notes

- Loss functions do NOT compute gradients manually
- Autograd handles gradient computation automatically
- Optimizers only consume `param.grad`
- Applying softmax twice is a common mistake (especially with Cross Entropy)

---

## Summary

| Loss Function | Primary Use |
|--------------|-------------|
| MSE | Regression |
| BCE | Binary classification |
| Cross Entropy | Multi-class classification |

This document maps directly to `loss_functions.py` 





