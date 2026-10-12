import torch
from src.models.mlp import MLP

model = MLP()
x = torch.randn(2, 197, 768)

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)