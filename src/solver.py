import os
import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.Metrics import Metrics
from src.utils.config import (
    MCC, ACCURACY, AUPRC, AUROC_, DICE_SCORE, F1_SCORE, JACCARD_INDEX,
    PRECISION, RECALL, SPECIFICITY
)


@dataclass
class FitResult:
    best_epoch: int
    best_val_dice: float
    best_val_loss: float
    best_train_loss: float


class Solver:
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        model_name: str = "Model",
        run_name: str = "run",
        save_dir: str = "../saved_models",
        save_each_epoch: bool = False,
    ) -> None:
        self.epochs = int(epochs)
        self.device = device

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_metrics = Metrics(self.device, 2)
        self.val_metrics = Metrics(self.device, 2)

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_loss_batch = []

        self.best_model = None
        self.best_epoch = -1
        self.best_val_dice = -1.0
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")

        self.model_name = model_name
        self.run_name = run_name
        self.save_dir = save_dir
        self.save_each_epoch = save_each_epoch

        # Early stopping (optional) — currently disabled
        # self.early_stop_patience = 3
        # self.early_stop_min_delta = 1e-4
        # self._no_improve_epochs = 0

    def fit(self) -> FitResult:
        for epoch in range(1, self.epochs + 1):
            print(f"{self.run_name} - Epoch {epoch}/{self.epochs}")

            self.train_metrics.reset()
            self.val_metrics.reset()

            batch_curve, train_epoch_loss = self._train_one_epoch(epoch)
            self.train_loss_batch.extend(batch_curve)
            self.train_loss_history.append(train_epoch_loss)

            val_epoch_loss = self._evaluate(epoch)
            self.val_loss_history.append(val_epoch_loss)

            train_agg = self.train_metrics.compute()
            val_agg = self.val_metrics.compute()

            # Scheduler step
            if self.scheduler is not None:
                # If using ReduceLROnPlateau, you'd call scheduler.step(val_epoch_loss)
                # otherwise scheduler.step() is typical.
                self.scheduler.step()

            # Save epoch model optionally
            if self.save_each_epoch:
                self._save_checkpoint(epoch, train_agg, val_agg, train_epoch_loss, val_epoch_loss)

            # Track best model by validation Dice
            if val_agg[DICE_SCORE] > self.best_val_dice:
                self.best_val_dice = float(val_agg[DICE_SCORE])
                self.best_val_loss = float(val_epoch_loss)
                self.best_train_loss = float(train_epoch_loss)
                self.best_epoch = epoch
                self.best_model = copy.deepcopy(self.model)
                self._save_best(epoch, train_agg, val_agg, train_epoch_loss, val_epoch_loss)

                # Early stopping bookkeeping (disabled)
                # self._no_improve_epochs = 0
            else:
                # Early stopping bookkeeping (disabled)
                # self._no_improve_epochs += 1
                pass

            print(
                f"Training   - Acc: {train_agg[ACCURACY]:.4f} | Dice: {train_agg[DICE_SCORE]:.4f} | "
                f"IoU: {train_agg[JACCARD_INDEX]:.4f} | Loss: {train_epoch_loss:.4f} | MCC: {train_agg[MCC]:.4f}"
            )
            print(
                f"Validation - Acc: {val_agg[ACCURACY]:.4f} | Dice: {val_agg[DICE_SCORE]:.4f} | "
                f"IoU: {val_agg[JACCARD_INDEX]:.4f} | Loss: {val_epoch_loss:.4f} | MCC: {val_agg[MCC]:.4f}"
            )
            print()

            # Early stopping (disabled)
            # if self._no_improve_epochs >= self.early_stop_patience:
            #     print(f"Early stopping: no improvement for {self.early_stop_patience} epochs.")
            #     break

        # Print final metrics summaries (end of training)
        final_train = self.train_metrics.compute()
        final_val = self.val_metrics.compute()
        print("=== Final epoch metrics ===")
        self._print_metrics(final_train, prefix="Train")
        self._print_metrics(final_val, prefix="Val")

        print("\n=== Best epoch (by Val Dice) ===")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Val Dice: {self.best_val_dice:.4f}")
        print(f"Best Train Loss: {self.best_train_loss:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")

        return FitResult(
            best_epoch=self.best_epoch,
            best_val_dice=self.best_val_dice,
            best_val_loss=self.best_val_loss,
            best_train_loss=self.best_train_loss,
        )

    def _train_one_epoch(self, epoch: int):
        self.model.train()

        loss_sum = 0.0
        total = 0
        running_avg = 0.0
        batch_curve = []

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
        for batch_idx, (x, y) in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            l = float(loss.detach().item())

            running_avg += l
            loss_sum += l * bs
            total += bs
            batch_curve.append(running_avg / (batch_idx + 1))

            metric_dict = self.train_metrics.update(logits, y)
            pbar.set_postfix(
                Loss=f"{batch_curve[-1]:.4f}",
                Acc=f"{metric_dict[ACCURACY]:.4f}",
                Dice=f"{metric_dict[DICE_SCORE]:.4f}",
                IoU=f"{metric_dict[JACCARD_INDEX]:.4f}",
                MCC=f"{metric_dict[MCC]:.4f}",
            )

        return batch_curve, (loss_sum / max(total, 1))

    @torch.inference_mode()
    def _evaluate(self, epoch: int):
        self.model.eval()

        loss_sum = 0.0
        total = 0
        running_avg = 0.0

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Validation")
        for batch_idx, (x, y) in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            bs = x.size(0)
            l = float(loss.detach().item())

            running_avg += l
            loss_sum += l * bs
            total += bs

            metric_dict = self.val_metrics.update(logits, y)
            pbar.set_postfix(
                Loss=f"{running_avg / (batch_idx + 1):.4f}",
                Acc=f"{metric_dict[ACCURACY]:.4f}",
                Dice=f"{metric_dict[DICE_SCORE]:.4f}",
                IoU=f"{metric_dict[JACCARD_INDEX]:.4f}",
                MCC=f"{metric_dict[MCC]:.4f}",
            )

        return loss_sum / max(total, 1)

    def _save_best(self, epoch, train_agg, val_agg, train_loss, val_loss):
        out_dir = os.path.join(self.save_dir, self.model_name, self.run_name)
        os.makedirs(out_dir, exist_ok=True)
        filename = (
            f"best_epoch_{epoch}"
            f"_TDice_{train_agg[DICE_SCORE]:.4f}"
            f"_VDice_{val_agg[DICE_SCORE]:.4f}"
            f"_TLoss_{train_loss:.4f}"
            f"_VLoss_{val_loss:.4f}.pt"
        )
        torch.save(self.model.state_dict(), os.path.join(out_dir, filename))

    def _save_checkpoint(self, epoch, train_agg, val_agg, train_loss, val_loss):
        out_dir = os.path.join(self.save_dir, self.model_name, self.run_name, "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        filename = (
            f"epoch_{epoch}"
            f"_VDice_{val_agg[DICE_SCORE]:.4f}"
            f"_VLoss_{val_loss:.4f}.pt"
        )
        torch.save(self.model.state_dict(), os.path.join(out_dir, filename))

    @staticmethod
    def _print_metrics(m: dict, prefix: str):
        # print the common ones you care about
        keys = [ACCURACY, DICE_SCORE, JACCARD_INDEX, F1_SCORE, PRECISION, RECALL, SPECIFICITY, MCC]
        line = " | ".join([f"{k}: {m[k]:.4f}" for k in keys if k in m])
        print(f"{prefix} - {line}")
