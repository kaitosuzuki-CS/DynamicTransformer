import torch
import torch.nn as nn

from ..components import FFN


class LinearBlock(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps

        self.layers = nn.ModuleList(
            [
                FFN(
                    in_features=hps.features[i],
                    out_features=hps.features[i + 1],
                    dropout=hps.dropout,
                )
                for i in range(len(hps.features) - 1)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out
