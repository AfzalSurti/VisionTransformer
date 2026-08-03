import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, embed_dim=768):
        super().__init__()

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        Q = self.q(x)

        K = self.k(x)

        V = self.v(x)

        attention_scores = Q @ K.transpose(-2, -1)

        attention_scores = attention_scores / math.sqrt(Q.shape[-1])

        attention_weights = self.softmax(attention_scores)

        output = attention_weights @ V

        return output