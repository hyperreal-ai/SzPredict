"""
Generate mock label arrays that match CHB-MIT Protocol 3 test-set class distribution.

Used for end-to-end eval-pipeline testing BEFORE wiring in real CHB-MIT data.
Once the real loader lands, this script becomes unnecessary — but it lets the
baselines and metrics module be validated independently.

CHB-MIT rough Protocol 3 test-set class distribution (from our internal runs):
  interictal: ~94%
  preictal:   ~4%
  ictal:      ~2%

Events: ~15 seizures across the 7 test patients.
Each seizure contributes 60 preictal windows (1s each, 5min window)
plus a variable ictal duration.

Usage:
    python scripts/make_mock_labels.py --out data/mock/ --n 10000
"""
import argparse
import numpy as np
from pathlib import Path


def generate_windows(labels: np.ndarray, n_channels: int = 23, n_samples: int = 256, seed: int = 2) -> np.ndarray:
    """Produce synthetic float32 windows [N, C, T] with class-dependent signal.

    Interictal: clean gaussian noise (σ=1).
    Preictal:   gaussian noise + low-amplitude 10-12 Hz oscillation (σ=1.2).
    Ictal:      gaussian noise + high-amplitude 3-8 Hz oscillation (σ=2.0).

    NOT realistic EEG — just separable enough that a simple CNN should learn
    something, so we can validate the training loop without real data.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    windows = rng.standard_normal((n, n_channels, n_samples), dtype=np.float32)
    t = np.arange(n_samples) / 256.0  # seconds

    for i, lbl in enumerate(labels):
        if lbl == 1:  # preictal
            freq = rng.uniform(10, 12)
            phase = rng.uniform(0, 2 * np.pi)
            amp = 0.6
            windows[i] *= 1.2
            windows[i] += (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
        elif lbl == 2:  # ictal
            freq = rng.uniform(3, 8)
            phase = rng.uniform(0, 2 * np.pi)
            amp = 1.5
            windows[i] *= 2.0
            windows[i] += (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)

    return windows


def generate(n_windows: int, n_events: int = 15, seed: int = 1):
    """Produce (labels, event_ids, window_times) synthetic arrays."""
    rng = np.random.default_rng(seed)

    labels = np.zeros(n_windows, dtype=np.int64)
    event_ids = -np.ones(n_windows, dtype=np.int64)
    window_times = np.arange(n_windows, dtype=np.float64)  # 1s per window

    # Distribute n_events across the timeline, each contributing 60 preictal + ~30 ictal
    seizure_span = 90  # preictal 60 + ictal ~30
    margin = 600
    positions = rng.choice(
        np.arange(margin, n_windows - seizure_span - margin),
        size=n_events, replace=False,
    )
    positions.sort()

    event_onset_times = {}
    for e, start in enumerate(positions):
        preictal_start = int(start)
        preictal_end = preictal_start + 60
        ictal_start = preictal_end
        ictal_len = int(rng.integers(15, 45))
        ictal_end = ictal_start + ictal_len

        labels[preictal_start:preictal_end] = 1
        labels[ictal_start:ictal_end] = 2
        event_ids[preictal_start:ictal_end] = e
        event_onset_times[e] = float(window_times[ictal_start])

    return labels, event_ids, window_times, event_onset_times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/mock")
    p.add_argument("--n", type=int, default=10000, help="number of windows")
    p.add_argument("--events", type=int, default=15, help="number of seizure events")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--with-windows", action="store_true",
                   help="also generate synthetic float32 windows [N,23,256] for CNN testing")
    p.add_argument("--channels", type=int, default=23)
    p.add_argument("--samples", type=int, default=256)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, event_ids, window_times, onset_times = generate(args.n, args.events, args.seed)

    np.save(out / "labels.npy", labels)
    np.save(out / "event_ids.npy", event_ids)
    np.save(out / "window_times.npy", window_times)

    if args.with_windows:
        windows = generate_windows(labels, args.channels, args.samples, seed=args.seed + 100)
        np.save(out / "windows.npy", windows)
        print(f"  windows.npy: {windows.shape} dtype={windows.dtype}")

    import json
    with open(out / "event_onset_times.json", "w") as f:
        json.dump(onset_times, f, indent=2)

    print(f"wrote {args.n} windows to {out}/")
    print(f"  interictal: {(labels == 0).sum()} ({(labels == 0).mean() * 100:.2f}%)")
    print(f"  preictal:   {(labels == 1).sum()} ({(labels == 1).mean() * 100:.2f}%)")
    print(f"  ictal:      {(labels == 2).sum()} ({(labels == 2).mean() * 100:.2f}%)")
    print(f"  events:     {args.events}")


if __name__ == "__main__":
    main()
