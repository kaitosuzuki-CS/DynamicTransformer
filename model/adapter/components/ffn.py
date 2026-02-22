import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._dropout = dropout

        self.norm = nn.BatchNorm1d(in_features)
        self.act = nn.SiLU()
        self.ffn = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.norm(x)
        out = self.act(out)
        out = self.ffn(out)
        out = self.dropout(out)

        return out
