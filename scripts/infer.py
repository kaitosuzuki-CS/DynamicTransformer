import argparse

import torch

from model import DynamicViT, SimpleViT
from utils import create_dataset, load_config

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a ViT classifier trained with a Contextual Bandit for Adaptive Depth"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["simplevit", "dynamicvit"],
        help="Model to test: (simplevit/dynamicvit)",
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
    parser.add_argument(
        "--ckpt-path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--classifier-ckpt-path",
        type=str,
        required=False,
        default=None,
        help="Path to the classifier checkpoint",
    )

    args = parser.parse_args()
    version = args.model
    model_config_path = args.model_config
    train_config_path = args.train_config
    ckpt_path = args.ckpt_path
    classifier_ckpt_path = args.classifier_ckpt_path

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    train_loader, val_loader = create_dataset(train_hps.data)  # type: ignore

    if version == "simplevit":
        model = SimpleViT(hps, train_hps, train_loader, val_loader, device)
        model.infer(ckpt_path)
    elif version == "dynamicvit":
        model = DynamicViT(hps, train_hps, train_loader, val_loader, device)
        model.infer(ckpt_path, classifier_ckpt_path)
