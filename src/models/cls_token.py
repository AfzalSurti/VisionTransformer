import torch
import torch.nn as nn


class CLSToken(nn.Module):

    def __init__(self, embed_dim=768):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

    def forward(self, x):

        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        return x