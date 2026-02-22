import os
from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

parent_dir = Path(__file__).resolve().parent.parent


class CustomDataset(Dataset):
    def __init__(self, data_dir, train, transform):
        self.data = MNIST(
            root=data_dir, train=train, transform=transform, download=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label, idx


def create_dataset(hps):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    data_dir = os.path.join(parent_dir, getattr(hps, "data_dir", "data"))

    train_dataset = CustomDataset(data_dir, train=True, transform=transform)
    val_dataset = CustomDataset(data_dir, train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(hps.train_bs),
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hps.val_bs),
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    return train_loader, val_loader
