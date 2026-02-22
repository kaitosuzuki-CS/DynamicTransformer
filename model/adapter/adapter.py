import os
from pathlib import Path

import torch
import torch.nn as nn

from .blocks import Encoder, LinearBlock

parent_dir = Path(__file__).resolve().parent.parent.parent


class Adapter(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps
        self._im_channels = hps.im_channels
        self._num_actions = hps.num_actions

        self.encoder = Encoder(hps.im_channels, hps.encoder)
        self.linear = LinearBlock(hps.linear)
        self.pred_head = nn.Sequential(
            nn.SiLU(), nn.Linear(hps.linear.features[-1], hps.num_actions)
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
            torch.load(ckpt_path)["adapter_state_dict"]
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

    def sample_action(self, x, epsilon):
        B, A = x.shape

        n = torch.rand(B, device=x.device) < epsilon

        greedy_actions = x.argmax(dim=1)
        random_actions = torch.randint(0, self._num_actions, size=(B,)).to(x.device)

        action = torch.where(n, random_actions, greedy_actions)

        return action

    def forward(self, x, epsilon=0):
        out = self.encoder(x)
        out = self.linear(out)
        out = self.pred_head(out)

        action = self.sample_action(out, epsilon)

        return out, action
