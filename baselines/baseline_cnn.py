"""
baseline_cnn — minimal 1D-CNN on raw windowed EEG.

Deliberately naive reference implementation. Uses ONLY standard components:
  - torch.nn.Conv1d
  - torch.nn.BatchNorm1d
  - torch.nn.MaxPool1d
  - torch.nn.Linear

No wavelets. No FFT. No transformer. No RWKV. The simplest thing that can
produce non-degenerate seizure classification on CHB-MIT.

Training practice baked in (the lesson from our unreleased internal runs):
  - EQUAL class weights (no 75x/300x reweighting that creates degenerate predictors)
  - Model selection on balanced accuracy, NOT validation loss
  - Simple Adam, conservative LR (1e-3), early stop on balanced-acc plateau

Expected on CHB-MIT Protocol 3:
  - Sensitivity: 30-50% (some real seizure signal)
  - Specificity: 85-95% (majority-class bias remains but doesn't dominate)
  - Balanced accuracy: 55-65% (beats 33% random-chance floor)
  - NOT degenerate

Community target: beat these numbers. Swap the CNN body for your own backbone,
keep the training practice, submit results.

Usage:
    # Train on prepared windows
    python -m baselines.baseline_cnn train \\
        --train-x data/train_windows.npy --train-y data/train_labels.npy \\
        --val-x data/val_windows.npy   --val-y data/val_labels.npy \\
        --out runs/cnn_v1

    # Eval against held-out test split
    python -m baselines.baseline_cnn eval \\
        --ckpt runs/cnn_v1/best.pt \\
        --test-x data/test_windows.npy --test-y data/test_labels.npy \\
        --test-events data/test_event_ids.npy \\
        --out results/baseline_cnn.json

Input data format:
    windows.npy: float32 array, shape [N, C, T]
                 N = number of windows, C = channels (default 23 for CHB-MIT),
                 T = samples per window (default 256 = 1 second at 256 Hz).
    labels.npy:  int array, shape [N], values in {0, 1, 2}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from szpredict.metrics import compute_all, balanced_accuracy, confusion_matrix


class SimpleCNN(nn.Module):
    """Minimal 1D-CNN. ~50k params. Intentionally underwhelming — a floor for real models."""

    def __init__(self, n_channels: int = 23, n_samples: int = 256, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        pooled_len = n_samples // 8
        self.fc1 = nn.Linear(64 * pooled_len, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)  # EQUAL class weights by default
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def train_cmd(args):
    device = torch.device(args.device)

    tx = np.load(args.train_x).astype(np.float32)
    ty = np.load(args.train_y).astype(np.int64)
    vx = np.load(args.val_x).astype(np.float32)
    vy = np.load(args.val_y).astype(np.int64)

    assert tx.ndim == 3, f"train_x must be [N, C, T], got {tx.shape}"
    n_channels, n_samples = tx.shape[1], tx.shape[2]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    model = SimpleCNN(n_channels, n_samples).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleCNN: {n_params:,} params ({n_channels}ch x {n_samples} samples)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_bal = 0.0
    best_epoch = -1
    patience = args.patience
    no_improve = 0

    log_lines = []
    t0 = time.time()
    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        v_preds, v_labels = eval_model(model, val_loader, device)
        cm = confusion_matrix(v_preds, v_labels)
        bal = balanced_accuracy(cm)

        line = f"epoch {epoch:3d}  train_loss {tr_loss:.4f}  val_bal_acc {bal*100:.2f}%"
        print(line); log_lines.append(line)

        if bal > best_bal:
            best_bal = bal
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "n_channels": n_channels,
                "n_samples": n_samples,
                "best_bal_acc": bal,
                "epoch": epoch,
            }, out_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stop: no val_bal_acc improvement for {patience} epochs")
                break

    dt = time.time() - t0
    summary = {
        "best_epoch": best_epoch,
        "best_val_bal_acc": best_bal,
        "training_time_sec": dt,
        "n_params": n_params,
    }
    with open(out_dir / "train_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
        f.write(f"\nsummary: {json.dumps(summary, indent=2)}\n")
    print(f"best epoch {best_epoch}, val_bal_acc {best_bal*100:.2f}%, {dt:.1f}s")


def eval_cmd(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    n_channels = ckpt["n_channels"]
    n_samples = ckpt["n_samples"]

    model = SimpleCNN(n_channels, n_samples).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tx = np.load(args.test_x).astype(np.float32)
    ty = np.load(args.test_y).astype(np.int64)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
        batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    preds, labels = eval_model(model, test_loader, device)

    event_ids = None
    if args.test_events:
        event_ids = np.load(args.test_events).astype(np.int64).ravel()

    metrics = compute_all(preds, labels, event_ids=event_ids)

    submission = {
        "benchmark_version": "0.1",
        "protocol": args.protocol,
        "model_name": "SimpleCNN",
        "model_description": "3-layer 1D-CNN on raw EEG windows. Equal class weights, balanced-acc selection.",
        "model_params": ckpt.get("n_params") or sum(p.numel() for p in model.parameters()),
        "training_time_hours": None,
        "hardware": str(device),
        **metrics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"wrote {out_path}")
    print(f"  sensitivity:  {metrics['sensitivity']*100:.2f}%")
    print(f"  specificity:  {metrics['specificity']*100:.2f}%")
    print(f"  balanced_acc: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"  degenerate:   {metrics['degenerate']['is_degenerate']}")


def main():
    parser = argparse.ArgumentParser(description="SzPredict SimpleCNN baseline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train-x", required=True)
    p_train.add_argument("--train-y", required=True)
    p_train.add_argument("--val-x", required=True)
    p_train.add_argument("--val-y", required=True)
    p_train.add_argument("--out", required=True)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--test-x", required=True)
    p_eval.add_argument("--test-y", required=True)
    p_eval.add_argument("--test-events", type=str, default=None)
    p_eval.add_argument("--out", required=True)
    p_eval.add_argument("--protocol", type=str, default="cross_patient_fixed")
    p_eval.add_argument("--batch-size", type=int, default=256)
    p_eval.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.cmd == "train":
        train_cmd(args)
    else:
        eval_cmd(args)


if __name__ == "__main__":
    main()
