import torch
import torch.nn as nn


class ConvolutionLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dropout
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dropout = dropout

        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.norm(x)
        out = self.act(out)
        out = self.conv(out)
        out = self.dropout(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_layers, dropout):
        super().__init__()

        self._channels = channels
        self._num_layers = num_layers
        self._dropout = dropout

        self.layers = nn.ModuleList(
            [
                ConvolutionLayer(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = out + layer(out)

        return out
