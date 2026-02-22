import argparse

import torch

from model import DynamicViT, SimpleViT
from utils import create_dataset, load_config

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a ViT classifier with a Contextual Bandit for Adaptive Depth"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["simplevit", "dynamicvit"],
        help="Model to train: (simplevit/dynamicvit)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Path to the training configuration file",
    )

    args = parser.parse_args()
    version = args.model
    model_config_path = args.model_config
    train_config_path = args.train_config

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    train_loader, val_loader = create_dataset(train_hps.data)  # type: ignore

    if version == "simplevit":
        model = SimpleViT(hps, train_hps, train_loader, val_loader, device)
    elif version == "dynamicvit":
        model = DynamicViT(hps, train_hps, train_loader, val_loader, device)

    model.train()
