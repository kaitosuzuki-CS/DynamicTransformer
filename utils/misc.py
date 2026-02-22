import copy
import os
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

parent_dir = Path(__file__).resolve().parent.parent


class HPS:
    def __init__(self, hps):
        for k, v in hps.items():
            if isinstance(v, dict):
                setattr(self, k, HPS(v))
            else:
                setattr(self, k, v)


def load_config(config_path):
    config_path = os.path.join(parent_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return HPS(config)


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.best_loss = float("inf")
        self.best_model = None
        self.stop = False

    def __call__(self, model, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                return self.best_model

        return None


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
