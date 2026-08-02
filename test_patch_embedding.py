import torch

from src.models.patch_embedding import PatchEmbedding

model = PatchEmbedding()

image = torch.randn(1,3,224,224)

output = model(image)

print(output.shape)