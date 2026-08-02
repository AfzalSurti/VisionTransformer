import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    ):
        super().__init__()

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x