import torch

from src.models.cls_token import CLSToken

x = torch.randn(2,196,768)

cls = CLSToken()

y = cls(x)

print(y.shape)