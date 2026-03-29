"""Generic training loop with early stopping, LR scheduling, and TensorBoard logging."""

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import Config
from models import BaseModel


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        config: Config,
        class_weights: Optional[np.ndarray] = None,
        device: str = "cuda",
        run_name: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config

        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32).to(device)
            if class_weights is not None
            else None
        )
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.warmup_epochs = config.training.warmup_epochs
        self.base_lr = config.training.learning_rate
        self._scheduler_deferred = config.training.scheduler == "one_cycle"
        self.scheduler = None if self._scheduler_deferred else self._build_scheduler(config)
        self.early_stopping = EarlyStopping(config.training.early_stopping_patience)

        name = run_name or f"multiscale_{config.data.decision_timeframe}"
        log_dir = os.path.join(config.export.output_dir, "runs", name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self._best_val_f1 = -1.0
        self._best_checkpoint_path: Optional[str] = None

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for x_5m, x_15m, x_1h, y in train_loader:
            x_5m = x_5m.to(self.device)
            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x_5m, x_15m, x_1h)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # OneCycleLR steps per batch, not per epoch
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        return {"loss": avg_loss, "accuracy": float(accuracy), "f1": float(f1)}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        from sklearn.metrics import precision_score, recall_score

        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []

        with torch.no_grad():
            for x_5m, x_15m, x_1h, y in val_loader:
                x_5m = x_5m.to(self.device)
                x_15m = x_15m.to(self.device)
                x_1h = x_1h.to(self.device)
                y = y.to(self.device)
                logits = self.model(x_5m, x_15m, x_1h)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = float(np.mean(all_preds == all_labels))
        precision = float(precision_score(all_labels, all_preds, zero_division=0))
        recall = float(recall_score(all_labels, all_preds, zero_division=0))
        f1 = float(f1_score(all_labels, all_preds, average="binary", zero_division=0))
        return {"loss": avg_loss, "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_prefix: str = "outputs/best_model",
        start_epoch: int = 0,
    ) -> Dict:
        os.makedirs(os.path.dirname(checkpoint_prefix) or ".", exist_ok=True)
        history = []

        # OneCycleLR needs steps_per_epoch — build it here
        if self._scheduler_deferred:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.base_lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.3,
                anneal_strategy="cos",
            )

        pbar = tqdm(range(start_epoch + 1, epochs + 1), desc="Training", unit="epoch")
        for epoch in pbar:
            # Linear warmup: scale LR from 0 to base_lr over warmup_epochs
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs and not isinstance(self.scheduler, OneCycleLR):
                warmup_lr = self.base_lr * epoch / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.writer.add_scalars("loss", {"train": train_metrics["loss"],
                                             "val": val_metrics["loss"]}, epoch)
            self.writer.add_scalars("f1", {"train": train_metrics["f1"],
                                           "val": val_metrics["f1"]}, epoch)
            self.writer.add_scalar("val/precision", val_metrics["precision"], epoch)
            self.writer.add_scalar("val/recall", val_metrics["recall"], epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", current_lr, epoch)

            # OneCycleLR steps per batch (in train_epoch), skip here
            if not isinstance(self.scheduler, OneCycleLR) and epoch > self.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            if val_metrics["f1"] > self._best_val_f1:
                self._best_val_f1 = val_metrics["f1"]
                self._best_checkpoint_path = f"{checkpoint_prefix}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": val_metrics["f1"],
                    "config": self.config,
                }, self._best_checkpoint_path)

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            best_marker = " *" if val_metrics["f1"] >= self._best_val_f1 else ""
            print(f"\nEpoch {epoch}/{epochs} | LR: {current_lr:.6f}"
                  f" | Train loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}"
                  f" | Val loss: {val_metrics['loss']:.4f}, P: {val_metrics['precision']:.4f},"
                  f" R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}{best_marker}")
            pbar.set_postfix({
                "train_loss": f"{train_metrics['loss']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
            })

            if self.early_stopping.step(val_metrics["loss"]):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.writer.close()
        return {
            "best_epoch": max(history, key=lambda h: h["val"]["f1"])["epoch"],
            "best_val_f1": self._best_val_f1,
            "checkpoint_path": self._best_checkpoint_path,
            "history": history,
        }

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model and optimizer state from checkpoint. Returns the epoch number."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        self._best_val_f1 = checkpoint.get("val_f1", -1.0)
        self._best_checkpoint_path = checkpoint_path
        print(f"Resumed from checkpoint: epoch {epoch}, val_f1={self._best_val_f1:.4f}")
        return epoch

    def _build_scheduler(self, config: Config):
        name = config.training.scheduler
        if name == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                patience=config.training.scheduler_patience,
                factor=config.training.scheduler_factor,
            )
        elif name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=config.training.epochs)
        elif name == "cosine_warmup":
            effective_epochs = config.training.epochs - config.training.warmup_epochs
            return CosineAnnealingLR(self.optimizer, T_max=max(effective_epochs, 1))
        elif name == "one_cycle":
            return None  # built in fit() after knowing steps_per_epoch
        elif name == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            raise ValueError(f"Unknown scheduler: {name!r}")
