"""
MLP Binary Classification on Breast Cancer (Sklearn built-in)

Goal: Simple PyTorch training/evaluation task with a scheduler feature (StepLR).
Self-verifiable: asserts validation metrics and exits non-zero on failure.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict:
    return {
        "task_id": "mlp_new_breastcancer_stepLR",
        "series": "Deep Learning (Simple)",
        "task_type": "classification",
        "dataset": "sklearn.datasets.load_breast_cancer",
        "model": "MLP",
        "features": ["StandardScaler", "StepLR"],
        "metrics": ["accuracy", "f1", "auc", "mse", "r2_like"],
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.2
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    step_size: int = 10
    gamma: float = 0.7


def make_dataloaders(cfg: Dict = None) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg or {}
    c = Config(**{**Config().__dict__, **cfg})

    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=c.test_size, random_state=c.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=c.batch_size, shuffle=False)
    return train_loader, val_loader


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: Dict = None) -> nn.Module:
    data = load_breast_cancer()
    in_dim = int(data.data.shape[1])
    return MLP(in_dim=in_dim)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict = None,
) -> Dict:
    cfg = cfg or {}
    c = Config(**{**Config().__dict__, **cfg})

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=c.step_size, gamma=c.gamma)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state = None

    for _epoch in range(c.epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)

        model.eval()
        with torch.no_grad():
            running = 0.0
            n = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                running += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            val_loss = running / max(n, 1)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    probs = []
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            yhat = torch.argmax(logits, dim=1)
            ys.append(yb.detach().cpu().numpy())
            probs.append(p.detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return y_true, y_pred, y_prob


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    y_true, y_pred, y_prob = predict(model, loader, device)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    # "Standard" regression-style metrics (protocol asks for MSE/R2).
    # For binary classification we compute them on probabilities vs labels.
    y_true_f = y_true.astype(np.float32)
    mse = float(np.mean((y_prob.astype(np.float32) - y_true_f) ** 2))
    denom = float(np.sum((y_true_f - float(np.mean(y_true_f))) ** 2)) + 1e-12
    r2_like = 1.0 - float(np.sum((y_true_f - y_prob.astype(np.float32)) ** 2) / denom)

    return {"accuracy": acc, "f1": f1, "auc": auc, "mse": mse, "r2": r2_like}


def save_artifacts(output_dir: str, artifacts: Dict) -> None:
    path = os.path.join(output_dir, "artifacts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)


def main() -> int:
    cfg = {}
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders(cfg)
    model = build_model(cfg).to(device)

    history = train(model, train_loader, val_loader, device, cfg)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    print("Train metrics:", train_metrics)
    print("Val metrics:", val_metrics)

    save_artifacts(
        OUTPUT_DIR,
        {
            "metadata": get_task_metadata(),
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
    )

    try:
        assert val_metrics["accuracy"] >= 0.90, f"val accuracy too low: {val_metrics['accuracy']:.3f}"
        assert val_metrics["f1"] >= 0.90, f"val F1 too low: {val_metrics['f1']:.3f}"
        assert np.isfinite(val_metrics["mse"]), "mse is not finite"
        print("PASS: quality thresholds met.")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

