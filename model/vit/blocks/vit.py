import torch
import torch.nn as nn

from ..components import Embedding, LinearLayer


class ViTLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout):
        super().__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._dropout = dropout

        self.norm_attn = nn.RMSNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout)

        self.norm_ffn = nn.RMSNorm(embed_dim)
        self.ffn = LinearLayer(embed_dim, hidden_dim, dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        _x = self.norm_attn(x)
        _x, _ = self.attn(_x, _x, _x, attn_mask=None)
        x = x + self.dropout_attn(_x)

        _x = self.norm_ffn(x)
        _x = self.ffn(_x)
        x = x + self.dropout_ffn(_x)

        return x


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, hps):
        super().__init__()

        self._embed_dim = embed_dim
        self._hps = hps

        self.layers = nn.ModuleList(
            [
                ViTLayer(
                    embed_dim=embed_dim,
                    hidden_dim=hps.hidden_dim,
                    num_heads=hps.num_heads,
                    dropout=hps.dropout,
                )
                for _ in range(hps.num_layers)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out


class ViT(nn.Module):
    def __init__(self, im_channels, embed_dim, hps):
        super().__init__()

        self._im_channels = im_channels
        self._embed_dim = embed_dim
        self._hps = hps

        self._max_timesteps = hps.max_timesteps

        self.emb = Embedding(im_channels, embed_dim, hps.embedding)
        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, hps.block) for _ in range(hps.max_timesteps)]
        )

    def forward(self, x, num_timesteps):
        out = self.emb(x)
        num_timesteps = (
            num_timesteps if num_timesteps is not None else self._max_timesteps
        )

        B, N, D = out.shape
        for t in range(num_timesteps):
            out = self.blocks[t](out)

        cls_token = out[:, 0, :]
        return cls_token, out
