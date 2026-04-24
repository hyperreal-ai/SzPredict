"""
baseline_majority — always predict interictal.

This is the degenerate-trap baseline. Demonstrates what a model that simply
'learned' the class prior looks like: high specificity (interictal is the
dominant class) and ZERO sensitivity to seizures.

The entire point of this baseline is to show that high accuracy numbers in
the seizure prediction literature can be achieved by simply predicting the
majority class. If your model's sensitivity is near zero, you've produced
this baseline in disguise.

Usage:
    python -m baselines.baseline_majority --labels path/to/labels.npy \\
        --out results/baseline_majority.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from szpredict.metrics import compute_all


def predict(n_windows: int) -> np.ndarray:
    return np.zeros(n_windows, dtype=np.int64)  # always interictal


def main():
    p = argparse.ArgumentParser(description="SzPredict majority-class (always-interictal) baseline")
    p.add_argument("--labels", type=str, required=True, help="path to .npy label array")
    p.add_argument("--event-ids", type=str, default=None, help="optional .npy event-id array")
    p.add_argument("--out", type=str, required=True, help="output submission JSON path")
    p.add_argument("--protocol", type=str, default="cross_patient_fixed")
    args = p.parse_args()

    labels = np.load(args.labels).astype(np.int64).ravel()
    preds = predict(len(labels))

    event_ids = None
    if args.event_ids:
        event_ids = np.load(args.event_ids).astype(np.int64).ravel()

    metrics = compute_all(preds, labels, event_ids=event_ids)

    submission = {
        "benchmark_version": "0.1",
        "protocol": args.protocol,
        "model_name": "Majority-Interictal",
        "model_description": "Always predicts interictal (degenerate trap demonstration)",
        "model_params": 0,
        "training_time_hours": 0.0,
        "hardware": "CPU only (no training)",
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
    print(f"  degenerate:   {metrics['degenerate']['is_degenerate']}  "
          f"(class: {metrics['degenerate']['single_class']})")


if __name__ == "__main__":
    main()
