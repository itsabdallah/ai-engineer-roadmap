"""
Loss Functions From Scratch (Module-Style)

This file implements common loss functions as callable classes,
similar to torch.nn loss modules, but simplified and explicit.

Loss functions:
- Measure how wrong predictions are
- Produce a scalar tensor
- Are the starting point of backpropagation
"""

import torch


class MSELoss:
    """
    Mean Squared Error Loss
    Used primarily for regression tasks.

    Formula:
        loss = mean((prediction - target) ** 2)
    """

    def __call__(self, prediction, target):
        # Difference between predicted values and ground truth
        diff = prediction - target

        # Square the difference
        squared_diff = diff ** 2

        # Return mean of squared differences (scalar tensor)
        return squared_diff.mean()


class BCELoss:
    """
    Binary Cross Entropy Loss
    Used for binary classification.

    Assumes prediction is already passed through sigmoid.
    """

    def __call__(self, prediction, target):
        epsilon = 1e-8  # prevents log(0)

        # Clamp predictions to avoid numerical instability
        prediction = torch.clamp(prediction, epsilon, 1 - epsilon)

        # BCE formula
        loss = -(target * torch.log(prediction) +
                 (1 - target) * torch.log(1 - prediction))

        return loss.mean()


class CrossEntropyLoss:
    """
    Cross Entropy Loss for multi-class classification.

    Assumptions:
    - prediction: raw logits (no softmax applied)
    - target: class indices (not one-hot)
    """

    def __call__(self, logits, target):
        # Apply log-softmax for numerical stability
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)

        # Select log-probabilities corresponding to correct classes
        batch_indices = torch.arange(target.shape[0])
        correct_log_probs = log_probs[batch_indices, target]

        # Negative log likelihood
        return -correct_log_probs.mean()
