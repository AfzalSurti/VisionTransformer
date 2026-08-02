import torch

from src.models.patch_embedding import PatchEmbedding
from src.models.positional_embedding import PositionEmbedding


patch = PatchEmbedding()

position = PositionEmbedding()

image = torch.randn(1,3,224,224)

x = patch(image)

print(x.shape)

x = position(x)

print(x.shape)