import torch
from activations import ReLU

def main():
    # Create input tensor
    x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)

    # Create activation
    relu = ReLU()

    # Forward pass
    y = relu(x)

    # Backward pass
    y.sum().backward()

    # Inspect results
    print("Input:", x)
    print("Output:", y)
    print("Gradient:", x.grad)

if __name__ == "__main__":
    main()
