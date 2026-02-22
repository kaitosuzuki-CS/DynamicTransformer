import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from utils import EarlyStopping, set_seeds

from .adapter import Adapter
from .vit import ViTClassifier

parent_dir = Path(__file__).resolve().parent.parent


class DynamicViT:
    def __init__(self, hps, train_hps, train_loader, val_loader, device):
        self._hps = hps
        self._adapter_hps = hps.adapter
        self._classifier_hps = hps.classifier
        self._train_hps = train_hps
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._init_hyperparameters()

        self.adapter = Adapter(hps.adapter)
        self.classifier = ViTClassifier(hps.classifier)

    def _init_hyperparameters(self):
        self.loss_hps = self._train_hps.loss
        self.optimizer_hps = self._train_hps.optimizer
        self.scheduler_hps = getattr(self._train_hps, "scheduler", None)
        self.early_stopping_hps = getattr(self._train_hps, "early_stopping", None)

        self.lr = float(self.optimizer_hps.lr)
        self.betas = tuple(map(float, self.optimizer_hps.betas))
        self.weight_decay = float(getattr(self.optimizer_hps, "weight_decay", 0))

        if self.scheduler_hps is not None:
            self.warmup_epochs = int(self.scheduler_hps.warmup_epochs)

        if self.early_stopping_hps is not None:
            self.patience = int(self.early_stopping_hps.patience)
            self.min_delta = float(self.early_stopping_hps.min_delta)

        self.eps = float(self.loss_hps.eps)
        self.eps_rate = float(self.loss_hps.eps_rate)
        self.eps_min = float(self.loss_hps.eps_min)
        self.eps_update_freq = int(self.loss_hps.eps_update_freq)
        self.lambda_n = float(self.loss_hps.lambda_n)

        self.max_timesteps = int(self._classifier_hps.max_timesteps)

        self.num_epochs = int(self._train_hps.num_epochs)
        self.accum_steps = int(getattr(self._train_hps, "accum_steps", 1))
        self.checkpoints_dir = os.path.join(
            parent_dir, str(self._train_hps.checkpoints_dir)
        )
        self.checkpoints_freq = int(self._train_hps.checkpoints_freq)

        self.use_probs = bool(getattr(self._train_hps, "use_probs", False))

        self.seed = int(getattr(self._train_hps, "seed", 42))

    def _init_training_scheme(self):
        optim = Adam(
            self.adapter.parameters(),
            lr=self.lr,
            betas=self.betas,  # type: ignore
            weight_decay=self.weight_decay,
        )

        scheduler = None
        if self.scheduler_hps is not None:
            num_steps_per_epoch = np.ceil(len(self._train_loader) / self.accum_steps)
            num_warmup_steps = int(self.warmup_epochs * num_steps_per_epoch)
            num_training_steps = int(self.num_epochs * num_steps_per_epoch)
            scheduler = get_cosine_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        early_stopping = None
        if self.early_stopping_hps is not None:
            early_stopping = EarlyStopping(
                patience=self.patience, min_delta=self.min_delta
            )

        return optim, scheduler, early_stopping

    def _init_weights(self):
        ckpt_path = str(self._classifier_hps.ckpt_path)

        self.adapter.init_weights()
        self.classifier.init_weights_with_ckpt(ckpt_path, freeze=True)

    def _init_weights_with_ckpt(
        self, adapter_ckpt_path, classifier_ckpt_path=None, freeze=True
    ):
        if classifier_ckpt_path is None:
            classifier_ckpt_path = str(self._classifier_hps.ckpt_path)

        self.adapter.init_weights_with_ckpt(adapter_ckpt_path, freeze)
        self.classifier.init_weights_with_ckpt(classifier_ckpt_path, freeze)

    def _move_to_device(self, device):
        self.adapter = self.adapter.to(device)
        self.classifer = self.classifier.to(device)

        print(f"Moved to {device}")

    def _eps_scheduler(self):
        self.eps = max(self.eps_min, self.eps * self.eps_rate)

    def train(self):
        set_seeds(self.seed)
        self._init_weights()
        self._move_to_device(self._device)
        self.classifier.eval()

        optim, scheduler, early_stopping = self._init_training_scheme()

        os.makedirs(self.checkpoints_dir, exist_ok=True)

        best_model = None
        num_updates = 0
        for epoch in range(1, self.num_epochs + 1):
            self.adapter.train()

            optim.zero_grad(set_to_none=True)
            train_loss = 0.0
            num_batches = 0
            for x, labels, _ in tqdm(self._train_loader, leave=False):
                num_batches += 1
                x, labels = x.to(self._device), labels.to(self._device)

                values, num_timesteps = self.adapter(x, epsilon=self.eps)

                B = values.shape[0]
                unique_steps = num_timesteps.unique(sorted=False)
                rewards = torch.zeros(B, device=self._device)
                with torch.no_grad():
                    for t in unique_steps:
                        idx = num_timesteps == t
                        timestep = t.item() + 1

                        logits = self.classifier(x[idx], timestep)

                        score = None
                        if self.use_probs:
                            logits = F.softmax(logits, dim=1)
                            logits = logits.gather(1, labels[idx].unsqueeze(1))
                            score = logits.squeeze(1)
                        else:
                            preds = logits.argmax(dim=1)
                            score = (preds == labels[idx]).float()

                        rewards[idx] = score - self.lambda_n * (
                            timestep / self.max_timesteps
                        )

                values = values.gather(1, num_timesteps.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(values, rewards.detach())
                loss = loss / self.accum_steps

                loss.backward()

                if num_batches % self.accum_steps == 0:
                    num_updates += 1
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                    if scheduler is not None:
                        scheduler.step()

                    if num_updates % self.eps_update_freq == 0:
                        self._eps_scheduler()

                train_loss += loss.item() * self.accum_steps

            if num_batches % self.accum_steps == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                self.adapter.eval()

                val_loss = 0.0

                for x, labels, _ in tqdm(self._val_loader, leave=False):
                    x, labels = x.to(self._device), labels.to(self._device)

                    values, num_timesteps = self.adapter(x, epsilon=1)

                    B = values.shape[0]
                    unique_steps = num_timesteps.unique(sorted=False)
                    rewards = torch.zeros(B, device=self._device)
                    for t in unique_steps:
                        idx = num_timesteps == t
                        timestep = t.item() + 1

                        logits = self.classifier(x[idx], timestep)

                        score = None
                        if self.use_probs:
                            logits = F.softmax(logits, dim=1)
                            logits = logits.gather(1, labels[idx].unsqueeze(1))
                            score = logits.squeeze(1)
                        else:
                            preds = logits.argmax(dim=1)
                            score = (preds == labels[idx]).float()

                        rewards[idx] = score - self.lambda_n * (
                            timestep / self.max_timesteps
                        )

                    values = values.gather(1, num_timesteps.unsqueeze(1)).squeeze(1)
                    loss = F.mse_loss(values, rewards)

                    val_loss += loss.item()

            train_loss /= len(self._train_loader)
            val_loss /= len(self._val_loader)

            print(f"----Epoch {epoch}----")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"LR: {optim.param_groups[0]['lr']:.6f}, Eps: {self.eps}")

            if epoch % self.checkpoints_freq == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "adapter_state_dict": self.adapter.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler is not None else None
                        ),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    f"{self.checkpoints_dir}/checkpoint_{epoch}.pt",
                )

            if early_stopping is not None:
                best_model = early_stopping(self.adapter, val_loss)
                if early_stopping.stop:
                    break

        if best_model is None:
            best_model = self.adapter

        torch.save(
            {
                "adapter_state_dict": best_model.state_dict(),
            },
            f"{self.checkpoints_dir}/final_model.pt",
        )

    def infer(self, adapter_ckpt_path, classifier_ckpt_path=None):
        self._init_weights_with_ckpt(
            adapter_ckpt_path, classifier_ckpt_path, freeze=True
        )
        self._move_to_device(self._device)

        self.adapter.eval()
        self.classifier.eval()

        total_timesteps = 0
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for x, labels, _ in tqdm(self._val_loader, leave=False):
                x, labels = x.to(self._device), labels.to(self._device)

                values, num_timesteps = self.adapter(x, epsilon=0)

                unique_steps = num_timesteps.unique(sorted=False)
                for t in unique_steps:
                    idx = num_timesteps == t
                    timestep = t.item() + 1

                    logits = self.classifier(x[idx], timestep)
                    preds = logits.argmax(dim=1)

                    num_correct += (preds == labels[idx]).sum().item()

                num_total += x.shape[0]
                total_timesteps += (num_timesteps + 1).sum().item()

        print(f"Accuracy: {num_correct / num_total:.6f}")
        print(f"Avg Timesteps: {total_timesteps / num_total:.4f}")
        print(f"Total Timesteps: {total_timesteps}")

        return total_timesteps
