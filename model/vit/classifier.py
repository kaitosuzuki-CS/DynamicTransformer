import os
from pathlib import Path

import torch
import torch.nn as nn

from .blocks import ViT

parent_dir = Path(__file__).resolve().parent.parent.parent


class ViTClassifier(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps
        self._embed_dim = hps.embed_dim
        self._im_channels = hps.im_channels

        self.vit = ViT(hps.im_channels, hps.embed_dim, hps)
        self.classifier = nn.Sequential(
            nn.RMSNorm(hps.embed_dim),
            nn.Linear(hps.embed_dim, hps.num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def init_weights_with_ckpt(self, ckpt_path, freeze=False):
        ckpt_path = os.path.join(parent_dir, ckpt_path)
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(ckpt_path)["classifier_state_dict"]
        )

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")
        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, x, num_timesteps=None):
        B, C, H, W = x.shape

        cls_token, _ = self.vit(x, num_timesteps)
        logits = self.classifier(cls_token)

        return logits
