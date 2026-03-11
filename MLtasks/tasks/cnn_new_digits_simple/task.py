"""
CNN Classification on Digits (Sklearn built-in)

Dataset: sklearn.datasets.load_digits (8x8 grayscale digits, 10 classes)
Goal: Simple CNN task that stays offline and self-verifies.
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
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata() -> Dict:
    return {
        "task_id": "cnn_new_digits_simple",
        "series": "Deep Learning (Simple)",
        "task_type": "classification",
        "dataset": "sklearn.datasets.load_digits",
        "model": "SmallCNN",
        "metrics": ["accuracy", "macro_f1", "mse", "r2_like"],
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
    epochs: int = 25
    lr: float = 2e-3
    weight_decay: float = 1e-4


def make_dataloaders(cfg: Dict = None) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg or {}
    c = Config(**{**Config().__dict__, **cfg})

    data = load_digits()
    X = data.images.astype(np.float32)  # (N, 8, 8) values 0..16
    y = data.target.astype(np.int64)

    # Normalize per-feature (flattened) with sklearn scaler for stability.
    X_flat = X.reshape(X.shape[0], -1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_flat, y, test_size=c.test_size, random_state=c.seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # Back to image tensors for CNN.
    X_train_img = X_train.reshape(-1, 1, 8, 8)
    X_val_img = X_val.reshape(-1, 1, 8, 8)

    train_ds = TensorDataset(torch.from_numpy(X_train_img), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_img), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=c.batch_size, shuffle=False)
    return train_loader, val_loader


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8->4
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4->2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(cfg: Dict = None) -> nn.Module:
    return SmallCNN(num_classes=10)


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
    optimizer = optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

    history = {"train_loss": [], "val_loss": []}
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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    preds = []
    probs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)
            yhat = torch.argmax(p, dim=1)
            ys.append(yb.detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())
            probs.append(p.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    return y_true, y_pred, y_prob


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    y_true, y_pred, y_prob = predict(model, loader, device)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    # Protocol-style MSE/R2 computed on one-hot targets vs predicted probs.
    k = int(np.max(y_true)) + 1
    y_onehot = np.eye(k, dtype=np.float32)[y_true]
    mse = float(np.mean((y_prob.astype(np.float32) - y_onehot) ** 2))
    denom = float(np.sum((y_onehot - float(np.mean(y_onehot))) ** 2)) + 1e-12
    r2_like = 1.0 - float(np.sum((y_onehot - y_prob.astype(np.float32)) ** 2) / denom)

    return {"accuracy": acc, "macro_f1": f1, "mse": mse, "r2": float(r2_like)}


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
        assert val_metrics["accuracy"] >= 0.93, f"val accuracy too low: {val_metrics['accuracy']:.3f}"
        assert val_metrics["macro_f1"] >= 0.92, f"val macro-F1 too low: {val_metrics['macro_f1']:.3f}"
        assert np.isfinite(val_metrics["mse"]), "mse is not finite"
        print("PASS: quality thresholds met.")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

