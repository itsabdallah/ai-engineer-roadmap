Autograd (Automatic Differentiation) in PyTorch
1. What is Autograd?
        Autograd is PyTorch’s automatic differentiation system.

        In simple terms:
        -Autograd automatically computes gradients (derivatives) for tensors involved in computations.

        Gradients tell us how much a value changes when another value changes. In machine learning, they are essential for training models.

        Without autograd, we would have to calculate derivatives manually, which is impractical for deep neural networks.

2. Why Gradients Matter in Training

        Training a neural network follows this loop:
            -Forward pass → compute predictions
            -Compute loss → measure error
            -Backward pass → compute gradients
            -Update parameters → reduce error

        Gradients answer the question:
        -How should each parameter change to reduce the loss?

        Autograd computes these gradients automatically using calculus rules (chain rule).

3. Enabling Gradient Tracking (requires_grad)
    By default, PyTorch tensors do not track gradients.

    To enable gradient tracking:
    x = torch.tensor(2.0, requires_grad=True)

-Explanation
    requires_grad=True tells PyTorch to:
        Track operations involving this tensor
        Build a computational graph
        Store information needed for backpropagation

You can verify it using:
print(x.requires_grad)

4. A Simple Autograd Example
        y = x ** 2

        Mathematically:
        y = x²


        If:
        x = 2


        Then:
        y = 4


        The derivative is:
        dy/dx = 2x

        Autograd stores this relationship internally.

5. The Computational Graph
        PyTorch builds a dynamic computational graph during execution.

        For the operation:
        y = x ** 2


        The graph conceptually looks like:
        x ──(square)──> y


        Each node knows:
        -What operation it represents
        -How to compute its derivative

        The graph exists only for the current forward pass.

6. Backpropagation with .backward()
        To compute gradients, we call:
        y.backward()


        This tells PyTorch to:
        -Start from y
        -Apply the chain rule
        -Compute gradients for all tensors with requires_grad=True

7. Accessing Gradients Using .grad
        After calling .backward():
        print(x.grad)


        This outputs:
        tensor(4.)


        Explanation:
        dy/dx = 2 × 2 = 4

        The gradient is stored in the .grad attribute of the tensor.

8. Gradient Accumulation
        By default, PyTorch accumulates gradients.

        Example:
        y.backward()
        y.backward()


        The gradients are added together instead of replaced.

        This is useful in training loops but can cause incorrect results if gradients are not reset properly.

9. Clearing Gradients

        To reset gradients manually:
            x.grad.zero_()


        In real training scenarios, this is usually handled by optimizers using:
            optimizer.zero_grad()

10. Disabling Gradient Tracking (torch.no_grad())

        Gradients are not always needed, for example:
        -During model evaluation
        -During inference
        -To save memory

        with torch.no_grad():
            y = x ** 2


        Inside this block:
        -No computational graph is created
        -No gradients are tracked

11. Detaching Tensors from the Graph

        You can remove a tensor from the computational graph using:
        y_detached = y.detach()


        This creates a new tensor that:
        -Shares the same data
        -Does not track gradients

        This is commonly used when converting tensors to NumPy or logging values.

12. Key Takeaways

        -Autograd tracks tensor operations dynamically
        -Gradients are computed using .backward()
        -Gradients accumulate unless cleared
        -torch.no_grad() disables tracking
        -.detach() removes tensors from the graph

        Autograd is the foundation of training neural networks in PyTorch.