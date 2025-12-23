# Tensor Fundamentals (PyTorch)

## Overview

This document records my understanding and hands-on verification of
core **tensor concepts** using PyTorch.

Tensors are the fundamental data structure in deep learning.
Everything in neural networks — inputs, weights, gradients, and outputs —
is represented as tensors.

This section builds the mathematical and practical intuition required
for neural networks, transformers, and large language models.

---

## 1. Tensor Creation & Shapes

A **tensor** is a generalization of numbers, vectors, and matrices
to arbitrary dimensions.

### Scalar
- **Rank (dimensions):** 0  
- **Shape:** `[]`  
- **Meaning:** A single number  

Example use:
- Loss values
- Learning rates
- Single numerical outputs

---

### Vector
- **Rank:** 1  
- **Shape:** `[n]`  
- **Meaning:** A list of numbers  

Example use:
- Feature vectors
- Word embeddings
- Bias terms in neural networks

---

### Matrix
- **Rank:** 2  
- **Shape:** `[rows, columns]`  
- **Meaning:** A table of numbers  

Example use:
- Weight matrices
- Linear layers
- Transformations between feature spaces

---

### 3D Tensor
- **Rank:** 3  
- **Shape:** `[batch, rows, columns]`  
- **Meaning:** A collection of matrices  

Example use:
- Batched training data
- Sequences of embeddings
- Image or token batches

Understanding tensor ranks is critical for avoiding shape errors
in neural network implementations.

---

## 2. Indexing & Slicing

Indexing and slicing allow us to **access or extract parts of tensors**.

### Examples:
- Access a single element using `[row, column]`
- Extract an entire row using `tensor[row]`
- Extract a column using `tensor[:, column]`
- Extract a sub-matrix using slicing ranges

Why this matters:
- Attention mechanisms rely on slicing
- Masking and padding depend on indexing
- Feature selection uses slicing heavily

Incorrect indexing is one of the most common sources of bugs
in deep learning code.

---

## 3. Broadcasting

**Broadcasting** is a mechanism that allows tensors of different shapes
to be combined in arithmetic operations.

Instead of manually copying data, PyTorch:
- Automatically expands smaller tensors
- Aligns dimensions when possible
- Performs operations efficiently

Example:
- Matrix shape: `[2, 3]`
- Vector shape: `[3]`
- The vector is applied to each row of the matrix

Why this matters:
- Bias addition in neural networks
- Normalization operations
- Efficient computation without extra memory usage

Broadcasting enables concise and high-performance code.

---

## 4. Matrix Multiplication

Matrix multiplication is the **core operation of neural networks**.

Two common ways to perform it:
- `torch.matmul(A, B)`
- `A @ B`

Important distinction:
- `*` → element-wise multiplication
- `@` / `matmul` → linear algebra operation

Why this matters:
- Neurons compute weighted sums using matrix multiplication
- Transformers rely on massive matrix multiplications
- Performance and correctness depend on understanding this difference

---

## 5. Device Placement (CPU / GPU)

Tensors can live on different devices:
- **CPU** (default)
- **GPU** (for accelerated computation)

Key concepts:
- Define a device (`cpu` or `cuda`)
- Move tensors using `.to(device)`
- Verify location using `.device`

Why this matters:
- Large models require GPUs
- Mixing CPU and GPU tensors causes runtime errors
- Efficient training depends on correct device placement

---

## Conclusion

This exercise confirms my ability to:

- Understand tensor dimensions and shapes
- Perform correct indexing and slicing
- Use broadcasting intentionally
- Distinguish matrix multiplication from element-wise operations
- Manage device placement explicitly

These skills form the foundation for:
- Automatic differentiation (autograd)
- Neural network layers
- Training loops
- Transformer architectures
