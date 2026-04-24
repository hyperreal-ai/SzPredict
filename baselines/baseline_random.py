"""
baseline_random — uniform random 3-class predictor.

Predicts interictal / preictal / ictal uniformly at random. Pure floor.
Any model worth its salt should beat this on ALL metrics. If it doesn't,
something's wrong.

Usage:
    python -m baselines.baseline_random --labels path/to/labels.npy \\
        --out results/baseline_random.json

The --labels file is an .npy of shape [N] with integer class labels
(0=interictal, 1=preictal, 2=ictal). This script only reads the SHAPE and
number of windows — it does not peek at label values. Predictions are
generated with a fixed seed for reproducibility.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from szpredict.metrics import compute_all


def predict(n_windows: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=3, size=n_windows, dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description="SzPredict random baseline")
    p.add_argument("--labels", type=str, required=True, help="path to .npy label array")
    p.add_argument("--event-ids", type=str, default=None, help="optional .npy event-id array")
    p.add_argument("--out", type=str, required=True, help="output submission JSON path")
    p.add_argument("--protocol", type=str, default="cross_patient_fixed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    labels = np.load(args.labels).astype(np.int64).ravel()
    preds = predict(len(labels), seed=args.seed)

    event_ids = None
    if args.event_ids:
        event_ids = np.load(args.event_ids).astype(np.int64).ravel()

    metrics = compute_all(preds, labels, event_ids=event_ids)

    submission = {
        "benchmark_version": "0.1",
        "protocol": args.protocol,
        "model_name": "Random-Uniform",
        "model_description": "Uniform random 3-class predictor (floor baseline)",
        "model_params": 0,
        "training_time_hours": 0.0,
        "hardware": "CPU only (no training)",
        "seed": args.seed,
        **metrics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"wrote {out_path}")
    print(f"  balanced_acc: {metrics['balanced_accuracy']*100:.2f}%")
    print(f"  sensitivity:  {metrics['sensitivity']*100:.2f}%")
    print(f"  specificity:  {metrics['specificity']*100:.2f}%")
    print(f"  degenerate:   {metrics['degenerate']['is_degenerate']}")


if __name__ == "__main__":
    main()
