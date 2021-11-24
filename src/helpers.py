"""
Misc helper classes and functions.
"""
import json
import os
from datetime import datetime
from typing import Optional, List
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def setup(args: dict, seed: int = 0) -> dict:
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return args


class EarlyStopper(object):
    def __init__(self, patience: int, filename: Optional[str] = None) -> None:
        if filename is None:
            # Name checkpoint based on time
            dt = datetime.now()
            filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
                dt.date(), dt.hour, dt.minute, dt.second
            )
            filename = os.path.join("/opt/ml/model", filename)

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def save_checkpoint(self, model: nn.Module) -> None:
        """Saves model when the metric on the validation set gets improved."""
        torch.save({"model_state_dict": model.state_dict()}, self.filename)

    def load_checkpoint(self, model: nn.Module) -> nn.Module:
        """Load model saved with early stopping."""
        model.load_state_dict(torch.load(self.filename)["model_state_dict"])

    def step(self, score: float, model: nn.Module) -> bool:
        if (self.best_score is None) or (score > self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                "EarlyStopping counter: {:d} out of {:d}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self) -> None:
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update for the result of an iteration
        Args:
            y_pred : float32 tensor
                Predicted molecule labels with shape (B, T),
                B for batch size and T for the number of tasks
            y_true : float32 tensor
                Ground truth molecule labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

    def roc_auc_score(self) -> List[float]:
        """Compute roc-auc score for each task.
        Returns: roc-auc score for all tasks
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_y_true = y_true[:, task].numpy()
            task_y_pred = y_pred[:, task].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores


def load_sagemaker_config(args: dict) -> dict:
    """
    Load SageMaker config from `/opt/ml/input/config/hyperparameters.json`
    """
    file_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            new_args = json.load(f)
            for k, v in new_args.items():
                if k not in args:
                    continue
                if isinstance(args[k], int):
                    v = int(v)
                if isinstance(args[k], float):
                    v = float(v)
                args[k] = v
    return args
