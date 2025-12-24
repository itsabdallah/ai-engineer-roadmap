import torch


def main():
    print("PyTorch Autograd Demo")
    print("-" * 50)

    # 1. Enabling gradient tracking
    x = torch.tensor(2.0, requires_grad=True)
    print("x:", x)
    print("x.requires_grad:", x.requires_grad)
    print()

    # 2. Simple computation
    y = x ** 2
    print("y = x ** 2")
    print("y:", y)
    print("y.requires_grad:", y.requires_grad)
    print()

    # 3. Backpropagation
    y.backward()
    print("After y.backward()")
    print("Gradient dy/dx (x.grad):", x.grad)
    print()

    # 4. Gradient accumulation demonstration
    y = x ** 2
    y.backward()
    print("After calling backward() again (accumulation)")
    print("Accumulated gradient (x.grad):", x.grad)
    print()

    # 5. Clearing gradients
    x.grad.zero_()
    print("After clearing gradients")
    print("x.grad:", x.grad)
    print()

    # 6. Disabling gradient tracking
    with torch.no_grad():
        y_no_grad = x ** 2

    print("Computation inside torch.no_grad()")
    print("y_no_grad:", y_no_grad)
    print("y_no_grad.requires_grad:", y_no_grad.requires_grad)
    print()

    # 7. Detaching tensors
    y = x ** 2
    y_detached = y.detach()

    print("Detached tensor")
    print("y.requires_grad:", y.requires_grad)
    print("y_detached.requires_grad:", y_detached.requires_grad)
    print("-" * 50)


if __name__ == "__main__":
    main()
