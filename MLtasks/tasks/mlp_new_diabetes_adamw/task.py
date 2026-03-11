"""
MLP Regression on Diabetes (Sklearn built-in)

Goal: Simple PyTorch regression task emphasizing a different optimizer (AdamW).
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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict:
    return {
        "task_id": "mlp_new_diabetes_adamw",
        "series": "Deep Learning (Simple)",
        "task_type": "regression",
        "dataset": "sklearn.datasets.load_diabetes",
        "model": "MLP",
        "optimizer": "AdamW",
        "metrics": ["mse", "rmse", "r2"],
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
    epochs: int = 200
    lr: float = 2e-3
    weight_decay: float = 2e-4
    patience: int = 30


def make_dataloaders(cfg: Dict = None) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg or {}
    c = Config(**{**Config().__dict__, **cfg})

    data = load_diabetes()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=c.test_size, random_state=c.seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=c.batch_size, shuffle=False)
    return train_loader, val_loader


class RegrMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: Dict = None) -> nn.Module:
    in_dim = int(load_diabetes().data.shape[1])
    return RegrMLP(in_dim=in_dim)


def _mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2)) + 1e-12
    return 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)


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
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

    history = {"train_mse": [], "val_mse": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(c.epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_mse = running / max(n, 1)

        val_m = evaluate(model, val_loader, device)
        val_mse = float(val_m["mse"])
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= c.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            ys.append(yb.detach().cpu().numpy())
            preds.append(pred)
    y_true = np.concatenate(ys, axis=0).astype(np.float32)
    y_pred = np.concatenate(preds, axis=0).astype(np.float32)
    return y_true, y_pred


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    y_true, y_pred = predict(model, loader, device)
    mse = _mse_np(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = _r2_np(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "r2": float(r2)}


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
        # Thresholds chosen to be achievable with a small MLP on this dataset.
        assert val_metrics["r2"] >= 0.25, f"val R2 too low: {val_metrics['r2']:.3f}"
        assert np.isfinite(val_metrics["mse"]), "mse is not finite"
        print("PASS: quality thresholds met.")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

