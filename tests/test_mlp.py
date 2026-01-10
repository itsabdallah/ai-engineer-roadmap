import torch
from mlp import MLP
from activations import ReLU

x = torch.randn(5, 2)
model = MLP(
    input_dim=2,
    hidden_dims=[8, 8],
    output_dim=1,
    activation=ReLU
)

y = model(x)
print(y.shape)

for p in model.parameters():
    print(p.is_leaf)

y.sum().backward()

for p in model.parameters():
    print(p.grad is not None)
