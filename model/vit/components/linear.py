import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._dropout = dropout

        self.in_layer = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        out = self.in_layer(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out
