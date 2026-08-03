import torch

from src.models.attention import SelfAttention

x = torch.randn(2,197,768)

attention = SelfAttention()

y = attention(x)

print(y.shape)