import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):

    def __init__(
        self,
        num_patches=196,
        embed_dim=768
    ):
        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                num_patches + 1,
                embed_dim
            )
        )

    def forward(self, x):

        return x + self.position_embedding[:,1:,:]