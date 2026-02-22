import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=128):
        super().__init__()

        self._embed_dim = embed_dim
        self._max_len = max_len

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        B, N, D = x.shape

        return self.pe[:, :N, :]  # type: ignore


class Embedding(nn.Module):
    def __init__(self, im_channels, embed_dim, hps):
        super().__init__()

        self._im_channels = im_channels
        self._embed_dim = embed_dim
        self._hps = hps

        self.patch_h, self.patch_w = hps.patch_size
        self.input_proj = nn.Sequential(
            nn.Linear(im_channels * self.patch_h * self.patch_w, embed_dim),
            nn.GELU(),
        )

        self.pad_value = nn.Parameter(torch.zeros(1))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pe = PositionalEmbedding(embed_dim, hps.max_len)

    def pad(self, x):
        B, C, H, W = x.shape

        h = np.ceil(H / self.patch_h) * self.patch_h
        w = np.ceil(W / self.patch_w) * self.patch_w

        pad_h = int(h - H)
        pad_w = int(w - W)

        if pad_h > 0:
            pad_tensor = self.pad_value.expand(B, C, pad_h, W)
            x = torch.cat([x, pad_tensor], dim=2)

        if pad_w > 0:
            pad_tensor = self.pad_value.expand(B, C, H + pad_h, W)
            x = torch.cat([x, pad_tensor], dim=3)

        return x

    def patchify(self, x):
        B, C, H, W = x.shape

        num_h = H // self.patch_h
        num_w = W // self.patch_w

        x = x.view(B, C, num_h, self.patch_h, num_w, self.patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, num_h * num_w, -1)

        return x

    def forward(self, x):
        B, C, H, W = x.shape

        out = self.pad(x)
        out = self.patchify(out)

        out = self.input_proj(out)
        out = torch.cat([self.cls_token.expand(B, -1, -1), out], dim=1)
        out = out + self.pe(out)

        return out
