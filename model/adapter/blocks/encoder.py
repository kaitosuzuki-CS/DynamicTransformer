import torch
import torch.nn as nn

from ..components import ConvolutionLayer, ResidualBlock


class EncoderLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        num_res_layers,
        dropout,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._num_res_layers = num_res_layers
        self._dropout = dropout

        self.in_layer = ConvolutionLayer(
            in_channels, out_channels, kernel_size, stride, padding, dropout
        )
        self.residual = ResidualBlock(out_channels, num_res_layers, dropout)

    def forward(self, x):
        out = self.in_layer(x)
        out = self.residual(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    in_channels=hps.channels[i],
                    out_channels=hps.channels[i + 1],
                    kernel_size=hps.kernel_size[i],
                    stride=hps.stride[i],
                    padding=hps.padding[i],
                    num_res_layers=hps.num_res_layers[i],
                    dropout=hps.dropout,
                )
                for i in range(len(hps.channels) - 1)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out


class Encoder(nn.Module):
    def __init__(self, im_channels, hps):
        super().__init__()

        self._im_channels = im_channels
        self._hps = hps

        self.in_conv = nn.Conv2d(
            im_channels,
            hps.channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.block = EncoderBlock(hps)

    def forward(self, x):
        out = self.in_conv(x)
        out = self.block(out)
        out = out.mean(dim=(2, 3))

        return out
