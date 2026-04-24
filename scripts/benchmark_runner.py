"""
SzPredict benchmark runner — CHB-MIT → labeled windows → submission pipeline
============================================================================

This script implements `spec/BENCHMARK_SPEC.md` end-to-end:

1. Load CHB-MIT EDF files via a per-subject summary.txt parser
2. Apply the preictal/interictal labelling rules from the spec
3. Split by patient according to `spec/splits.json` (Protocol 3)
4. Emit labels.npy + event_ids.npy + window_times.npy + event_onset_times.json
5. Optionally emit windows.npy (raw EEG — large, needed for training)

Zero proprietary components. Pure spec implementation. Swap the baseline model
(or your own architecture) for the downstream training / eval step.

Usage
-----
Prepare a single split for Protocol 3:

    python scripts/benchmark_runner.py prepare \\
        --chb-mit-dir data/chb-mit \\
        --out data/chbmit_p3 \\
        --protocol 3 --split train \\
        --window-seconds 1 --include-windows

Prepare all splits at once:

    python scripts/benchmark_runner.py prepare-all \\
        --chb-mit-dir data/chb-mit \\
        --out data/chbmit_p3 \\
        --protocol 3 --window-seconds 1 --include-windows

Run metrics on a predictions file:

    python scripts/benchmark_runner.py score \\
        --predictions my_preds.npy \\
        --labels data/chbmit_p3/test/labels.npy \\
        --event-ids data/chbmit_p3/test/event_ids.npy \\
        --out results/my_model_p3.json \\
        --model-name MyModel --protocol cross_patient_fixed

Dependencies: numpy, mne>=1.0 (for EDF loading).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from szpredict.metrics import compute_all

log = logging.getLogger("szpredict.benchmark_runner")


# ============================================================================
# Spec constants (mirror spec/BENCHMARK_SPEC.md — edit there, not here)
# ============================================================================

FS_HZ = 256                      # CHB-MIT native sampling rate
PREICTAL_SECONDS = 300           # 5 minutes before seizure onset → label=1
POSTICTAL_EXCLUDE_SECONDS = 300  # 5 minutes after offset → excluded
MIN_INTER_SEIZURE_SECONDS = 600  # <10 min between seizures → whole gap excluded

LABEL_INTERICTAL = 0
LABEL_PREICTAL = 1
LABEL_ICTAL = 2
LABEL_EXCLUDED = -1              # sentinel for windows to drop before saving


# ============================================================================
# CHB-MIT summary.txt parser
# ============================================================================

@dataclass
class Seizure:
    """A single annotated ictal event within one EDF file."""
    start_s: float       # seconds from start of the containing EDF file
    end_s: float


@dataclass
class EdfEntry:
    """Metadata for one EDF file, parsed from chbXX-summary.txt."""
    filename: str
    file_start_s: Optional[float]  # absolute seconds from some subject-level reference, if derivable
    file_duration_s: Optional[float]
    seizures: List[Seizure] = field(default_factory=list)

    @property
    def num_seizures(self) -> int:
        return len(self.seizures)


@dataclass
class SubjectIndex:
    """All EDF entries + annotations for one CHB-MIT subject."""
    subject_id: str              # e.g. "chb01"
    channels: List[str]
    entries: List[EdfEntry] = field(default_factory=list)


def _parse_hhmmss(s: str) -> Optional[float]:
    """Parse 'HH:MM:SS' → seconds. CHB-MIT sometimes wraps past 24:00; accept."""
    m = re.match(r"^\s*(\d{1,3}):(\d{2}):(\d{2})\s*$", s)
    if not m:
        return None
    h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mm * 60 + ss


def parse_chbmit_summary(summary_path: Path) -> SubjectIndex:
    """
    Parse a chbXX-summary.txt into a structured SubjectIndex.

    The CHB-MIT summary format is documented at:
        https://physionet.org/content/chbmit/1.0.0/chb-mit-README.txt
    It is a plain-text file. Per-file blocks look like:

        File Name: chb01_03.edf
        File Start Time: 13:43:04
        File End Time: 14:43:04
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds

    Some subjects have multiple seizures per file; each pair of
    Seizure Start/End lines is taken in order.
    """
    subject_id = summary_path.parent.name  # chbXX
    channels: List[str] = []
    entries: List[EdfEntry] = []
    current_entry: Optional[EdfEntry] = None
    current_seizure_start: Optional[float] = None

    with open(summary_path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Channels block
            m = re.match(r"^Channel\s+\d+:\s*(.+)$", line)
            if m:
                channels.append(m.group(1).strip())
                continue

            # File name — starts a new entry
            m = re.match(r"^File Name:\s*(.+)$", line)
            if m:
                if current_entry is not None:
                    entries.append(current_entry)
                current_entry = EdfEntry(
                    filename=m.group(1).strip(),
                    file_start_s=None,
                    file_duration_s=None,
                )
                current_seizure_start = None
                continue

            if current_entry is None:
                continue

            # File start/end time (optional, used to derive duration)
            m = re.match(r"^File Start Time:\s*(.+)$", line)
            if m:
                current_entry.file_start_s = _parse_hhmmss(m.group(1))
                continue
            m = re.match(r"^File End Time:\s*(.+)$", line)
            if m:
                end = _parse_hhmmss(m.group(1))
                if end is not None and current_entry.file_start_s is not None:
                    dur = end - current_entry.file_start_s
                    # Handle midnight wrap
                    if dur < 0:
                        dur += 24 * 3600
                    current_entry.file_duration_s = float(dur)
                continue

            # Seizure timestamps
            m = re.match(r"^Seizure\s*(?:\d+\s+)?Start Time:\s*([\d.]+)\s*seconds?$", line, re.IGNORECASE)
            if m:
                current_seizure_start = float(m.group(1))
                continue
            m = re.match(r"^Seizure\s*(?:\d+\s+)?End Time:\s*([\d.]+)\s*seconds?$", line, re.IGNORECASE)
            if m and current_seizure_start is not None:
                current_entry.seizures.append(
                    Seizure(start_s=current_seizure_start, end_s=float(m.group(1)))
                )
                current_seizure_start = None
                continue

            # "Number of Seizures in File: N" — informational; we derive count from parsed seizures
            # (don't enforce, since the text varies slightly across subjects)

    if current_entry is not None:
        entries.append(current_entry)

    return SubjectIndex(subject_id=subject_id, channels=channels, entries=entries)


# ============================================================================
# EDF loading (via MNE)
# ============================================================================

def load_edf_windows(
    edf_path: Path,
    window_samples: int,
    fs: int = FS_HZ,
    channels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Load one EDF and return non-overlapping windows.

    Returns
    -------
    windows : np.ndarray [N, C, T]  float32
    duration_s : float              length of the original recording in seconds
    """
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "benchmark_runner requires `mne` for EDF loading. "
            "Install with: pip install mne"
        ) from e

    # mne.io.read_raw_edf is verbose; silence it
    with _silence_mne():
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

    if int(round(raw.info["sfreq"])) != fs:
        log.warning(f"{edf_path.name}: sample rate {raw.info['sfreq']} != {fs}, resampling")
        raw.resample(fs)

    if channels is not None:
        # Keep only requested channels (intersection; skip missing)
        available = [ch for ch in channels if ch in raw.ch_names]
        if available:
            raw.pick(available)

    data = raw.get_data().astype(np.float32)  # [C, T_total]
    total_samples = data.shape[1]
    n_windows = total_samples // window_samples
    trimmed = data[:, : n_windows * window_samples]
    windows = trimmed.reshape(data.shape[0], n_windows, window_samples).transpose(1, 0, 2)
    duration_s = total_samples / fs
    return windows.astype(np.float32), duration_s


class _silence_mne:
    """Context manager to suppress mne.io verbose output."""
    def __enter__(self):
        import mne
        self._prev_level = mne.set_log_level("ERROR", return_old_level=True)
        return self

    def __exit__(self, *exc):
        import mne
        mne.set_log_level(self._prev_level)


# ============================================================================
# Spec-rule label generator
# ============================================================================

def label_windows_for_file(
    n_windows: int,
    window_seconds: float,
    seizures: List[Seizure],
    event_id_base: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply spec labelling rules to a single EDF's windows.

    Returns
    -------
    labels    : [n_windows] int64 — values in {-1, 0, 1, 2}. -1 = excluded.
    event_ids : [n_windows] int64 — per-seizure event id (-1 for non-seizure windows)

    Rules
    -----
    For each window w at time [t, t+window_seconds):
      - If overlaps any ictal range  [s.start_s, s.end_s)   → ictal (2), event_id = s
      - elif w overlaps preictal window [s.start_s - 300, s.start_s)  → preictal (1), event_id = s
      - elif w overlaps post-ictal exclusion [s.end_s, s.end_s + 300)  → excluded (-1)
      - elif w falls in inter-seizure gap where prev seizure ended < 600s ago AND next seizure starts
            within another <600s → excluded (-1, entire short gap dropped)
      - else → interictal (0), event_id = -1
    """
    labels = np.full(n_windows, LABEL_INTERICTAL, dtype=np.int64)
    event_ids = np.full(n_windows, -1, dtype=np.int64)

    if not seizures:
        return labels, event_ids

    # Sort seizures by start time (should already be — defensive)
    seizures_sorted = sorted(seizures, key=lambda s: s.start_s)

    # Mark ictal + preictal + post-ictal exclusion in one pass
    for i, s in enumerate(seizures_sorted):
        ev = event_id_base + i

        pre_start = s.start_s - PREICTAL_SECONDS
        pre_end = s.start_s
        post_start = s.end_s
        post_end = s.end_s + POSTICTAL_EXCLUDE_SECONDS

        for w in range(n_windows):
            w_start = w * window_seconds
            w_end = w_start + window_seconds

            # Ictal overlap wins
            if w_end > s.start_s and w_start < s.end_s:
                labels[w] = LABEL_ICTAL
                event_ids[w] = ev
                continue

            # Preictal (only if not already labelled ictal)
            if labels[w] == LABEL_INTERICTAL and w_end > pre_start and w_start < pre_end:
                labels[w] = LABEL_PREICTAL
                event_ids[w] = ev
                continue

            # Post-ictal exclusion
            if labels[w] == LABEL_INTERICTAL and w_end > post_start and w_start < post_end:
                labels[w] = LABEL_EXCLUDED

    # Short-gap exclusion: if two consecutive seizures are <MIN_INTER_SEIZURE_SECONDS apart,
    # drop the entire inter-seizure gap to avoid confounded labelling.
    for i in range(len(seizures_sorted) - 1):
        gap = seizures_sorted[i + 1].start_s - seizures_sorted[i].end_s
        if gap < MIN_INTER_SEIZURE_SECONDS:
            gap_start = seizures_sorted[i].end_s
            gap_end = seizures_sorted[i + 1].start_s
            for w in range(n_windows):
                w_start = w * window_seconds
                w_end = w_start + window_seconds
                if w_end > gap_start and w_start < gap_end:
                    labels[w] = LABEL_EXCLUDED

    return labels, event_ids


# ============================================================================
# Subject-level assembly
# ============================================================================

def prepare_subject(
    subject_dir: Path,
    window_seconds: float,
    include_windows: bool,
    event_id_base: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, float], Optional[np.ndarray]]:
    """
    Produce labelled windows for one CHB-MIT subject, concatenated across all EDF files.

    Returns
    -------
    labels        [N]
    event_ids     [N]  (global across subject)
    window_times  [N]  (seconds from subject start; concatenated)
    event_onset_times  {event_id: seconds-from-subject-start}
    windows       [N, C, T] or None if include_windows=False
    """
    summary_path = subject_dir / f"{subject_dir.name}-summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    index = parse_chbmit_summary(summary_path)

    window_samples = int(round(window_seconds * FS_HZ))
    all_labels: List[np.ndarray] = []
    all_event_ids: List[np.ndarray] = []
    all_window_times: List[np.ndarray] = []
    all_windows: List[np.ndarray] = []
    event_onset_times: Dict[int, float] = {}

    subject_elapsed_s = 0.0  # running clock across the concatenated subject recording
    event_counter = event_id_base

    for entry in index.entries:
        edf_path = subject_dir / entry.filename
        if not edf_path.exists():
            log.warning(f"Missing EDF {edf_path}, skipping")
            continue

        if include_windows:
            windows, duration_s = load_edf_windows(edf_path, window_samples, channels=index.channels)
            n = windows.shape[0]
        else:
            # Even without saving windows, we need duration_s to size labels
            if entry.file_duration_s is not None:
                duration_s = entry.file_duration_s
            else:
                # Fall back: quick EDF header read for duration
                try:
                    import mne
                    with _silence_mne():
                        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
                    duration_s = raw.n_times / raw.info["sfreq"]
                except Exception as e:
                    log.warning(f"Could not determine duration for {edf_path.name}: {e}")
                    continue
            n = int(duration_s // window_seconds)
            windows = None

        labels, event_ids = label_windows_for_file(
            n_windows=n,
            window_seconds=window_seconds,
            seizures=entry.seizures,
            event_id_base=event_counter,
        )

        # Capture onset times for events in this file (absolute from subject start)
        for i, sz in enumerate(sorted(entry.seizures, key=lambda s: s.start_s)):
            ev = event_counter + i
            event_onset_times[ev] = subject_elapsed_s + sz.start_s
        event_counter += len(entry.seizures)

        window_times = np.arange(n, dtype=np.float64) * window_seconds + subject_elapsed_s

        all_labels.append(labels)
        all_event_ids.append(event_ids)
        all_window_times.append(window_times)
        if include_windows and windows is not None:
            all_windows.append(windows)

        subject_elapsed_s += duration_s

    if not all_labels:
        raise RuntimeError(f"No usable EDF files found for {subject_dir}")

    labels = np.concatenate(all_labels)
    event_ids = np.concatenate(all_event_ids)
    window_times = np.concatenate(all_window_times)
    windows = np.concatenate(all_windows, axis=0) if (include_windows and all_windows) else None

    return labels, event_ids, window_times, event_onset_times, windows


# ============================================================================
# Split application + multi-subject assembly
# ============================================================================

def load_splits_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def prepare_split(
    chb_mit_dir: Path,
    subject_ids: List[str],
    window_seconds: float,
    include_windows: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], Optional[np.ndarray]]:
    """
    Assemble all subjects in a split into a single concatenated set of arrays.
    Event IDs are globally unique across the split.
    """
    all_labels = []
    all_event_ids = []
    all_window_times = []
    all_windows: List[np.ndarray] = []
    all_onset_times: Dict[str, float] = {}
    time_offset = 0.0
    event_counter = 0

    for sid in subject_ids:
        subject_dir = chb_mit_dir / sid
        if not subject_dir.exists():
            log.warning(f"Subject directory missing: {subject_dir}, skipping")
            continue

        log.info(f"Preparing {sid} ...")
        labels, event_ids, window_times, onset_times, windows = prepare_subject(
            subject_dir=subject_dir,
            window_seconds=window_seconds,
            include_windows=include_windows,
            event_id_base=event_counter,
        )

        # Offset window times so concatenation across subjects is monotonic
        window_times = window_times + time_offset
        subject_duration = window_times[-1] + window_seconds - time_offset if len(window_times) else 0

        # Offset event onset times similarly; prefix event id with subject for readability
        for ev, t in onset_times.items():
            all_onset_times[f"{sid}:{ev}"] = t + time_offset

        time_offset += subject_duration
        event_counter = int(event_ids.max()) + 1 if (event_ids >= 0).any() else event_counter

        # Drop excluded windows (label == -1) to produce a clean dataset
        mask = labels != LABEL_EXCLUDED
        all_labels.append(labels[mask])
        all_event_ids.append(event_ids[mask])
        all_window_times.append(window_times[mask])
        if include_windows and windows is not None:
            all_windows.append(windows[mask])

    if not all_labels:
        raise RuntimeError(f"No subjects produced data for split: {subject_ids}")

    labels = np.concatenate(all_labels)
    event_ids = np.concatenate(all_event_ids)
    window_times = np.concatenate(all_window_times)
    windows = np.concatenate(all_windows, axis=0) if all_windows else None

    return labels, event_ids, window_times, all_onset_times, windows


# ============================================================================
# Save helpers
# ============================================================================

def save_split_artifacts(
    out_dir: Path,
    labels: np.ndarray,
    event_ids: np.ndarray,
    window_times: np.ndarray,
    event_onset_times: Dict[str, float],
    windows: Optional[np.ndarray],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "event_ids.npy", event_ids)
    np.save(out_dir / "window_times.npy", window_times)
    with open(out_dir / "event_onset_times.json", "w") as f:
        json.dump(event_onset_times, f, indent=2)
    if windows is not None:
        np.save(out_dir / "windows.npy", windows)

    total_gb = sum(p.stat().st_size for p in out_dir.iterdir()) / (1024 ** 3)
    summary = {
        "n_windows": int(len(labels)),
        "interictal_count": int((labels == LABEL_INTERICTAL).sum()),
        "preictal_count": int((labels == LABEL_PREICTAL).sum()),
        "ictal_count": int((labels == LABEL_ICTAL).sum()),
        "unique_events": int(len(set(int(e) for e in event_ids if e >= 0))),
        "total_disk_gb": round(total_gb, 3),
        "includes_windows": windows is not None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================================================================
# CLI
# ============================================================================

def _find_protocol_key(splits: Dict, protocol: int) -> str:
    """Look up the full protocol key in splits.json by leading 'protocol_N'."""
    prefix = f"protocol_{protocol}"
    for k in splits:
        if k.startswith(prefix) and isinstance(splits[k], dict):
            return k
    raise SystemExit(
        f"splits.json has no key starting with {prefix}. "
        f"Found top-level keys: {[k for k in splits if isinstance(splits[k], dict)]}"
    )


def cmd_prepare(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = load_splits_json(Path(args.splits))
    protocol_key = _find_protocol_key(splits, args.protocol)
    split_dict = splits[protocol_key]
    if args.split not in split_dict:
        raise SystemExit(
            f"{protocol_key} does not define split '{args.split}'. "
            f"Available splits: {[k for k in split_dict if isinstance(split_dict[k], list)]}"
        )

    subjects = split_dict[args.split]
    log.info(f"Preparing protocol {args.protocol} split '{args.split}' over {len(subjects)} subjects")

    labels, event_ids, window_times, onset_times, windows = prepare_split(
        chb_mit_dir=Path(args.chb_mit_dir),
        subject_ids=subjects,
        window_seconds=args.window_seconds,
        include_windows=args.include_windows,
    )

    summary = save_split_artifacts(
        out_dir=Path(args.out),
        labels=labels,
        event_ids=event_ids,
        window_times=window_times,
        event_onset_times=onset_times,
        windows=windows,
    )
    log.info(f"Done: {json.dumps(summary, indent=2)}")


def cmd_prepare_all(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = load_splits_json(Path(args.splits))
    protocol_key = _find_protocol_key(splits, args.protocol)

    out_root = Path(args.out)
    for split_name, subjects in splits[protocol_key].items():
        if not isinstance(subjects, list):
            # Skip rationale / method / notes string fields
            continue
        log.info(f"=== Split: {split_name} ({len(subjects)} subjects) ===")
        labels, event_ids, window_times, onset_times, windows = prepare_split(
            chb_mit_dir=Path(args.chb_mit_dir),
            subject_ids=subjects,
            window_seconds=args.window_seconds,
            include_windows=args.include_windows,
        )
        split_out = out_root / split_name
        summary = save_split_artifacts(
            out_dir=split_out,
            labels=labels,
            event_ids=event_ids,
            window_times=window_times,
            event_onset_times=onset_times,
            windows=windows,
        )
        log.info(f"Split '{split_name}': {json.dumps(summary, indent=2)}")


def cmd_score(args):
    """Compute full metric suite on a predictions file against ground-truth labels."""
    predictions = np.load(args.predictions).astype(np.int64).ravel()
    labels = np.load(args.labels).astype(np.int64).ravel()

    event_ids = None
    window_times = None
    event_onset_times = None

    if args.event_ids:
        event_ids = np.load(args.event_ids).astype(np.int64).ravel()
    if args.window_times:
        window_times = np.load(args.window_times).astype(np.float64).ravel()
    if args.event_onset_times:
        with open(args.event_onset_times) as f:
            raw = json.load(f)
        # Accept either {ev_id: t} or {str_key: t}
        event_onset_times = {}
        for k, v in raw.items():
            # Keys in our save format look like "chb01:3"; score function expects int event ids.
            # Try to coerce; if it's a "subject:local_id" format, keep the hash-derived id consistent with event_ids.
            try:
                event_onset_times[int(k)] = float(v)
            except (ValueError, TypeError):
                # Not a plain int — user should pass event_ids that match keys OR preprocess
                pass

    metrics = compute_all(
        predictions=predictions,
        labels=labels,
        event_ids=event_ids,
        window_times=window_times,
        event_onset_times=event_onset_times,
    )

    submission = {
        "benchmark_version": "0.1",
        "protocol": args.protocol,
        "model_name": args.model_name,
        "model_description": args.model_description or "",
        "model_params": args.model_params,
        "training_time_hours": args.training_time_hours,
        "hardware": args.hardware or "",
        **metrics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Wrote submission: {out_path}")
    print(f"  sensitivity: {metrics['sensitivity'] * 100:.2f}%")
    print(f"  specificity: {metrics['specificity'] * 100:.2f}%")
    print(f"  balanced_accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  degenerate: {metrics['degenerate']['is_degenerate']}")


def main():
    parser = argparse.ArgumentParser(
        description="SzPredict benchmark runner: CHB-MIT → labelled windows → metrics",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- prepare --
    p_prep = sub.add_parser("prepare", help="Prepare one split (train, val, test) from CHB-MIT")
    p_prep.add_argument("--chb-mit-dir", required=True, help="Path to CHB-MIT dataset root")
    p_prep.add_argument("--out", required=True, help="Output directory for labels.npy etc.")
    p_prep.add_argument("--protocol", type=int, default=3, choices=[1, 2, 3],
                        help="Protocol number (default 3)")
    p_prep.add_argument("--split", required=True, help="Split name as in splits.json (e.g. train/val/test)")
    p_prep.add_argument("--window-seconds", type=float, default=1.0,
                        help="Window length in seconds (default 1.0)")
    p_prep.add_argument("--include-windows", action="store_true",
                        help="Also save raw windows.npy (large — needed for CNN training)")
    p_prep.add_argument("--splits", default=str(Path(__file__).resolve().parents[1] / "spec" / "splits.json"),
                        help="Path to splits.json (default: spec/splits.json)")
    p_prep.set_defaults(func=cmd_prepare)

    # -- prepare-all --
    p_all = sub.add_parser("prepare-all", help="Prepare all splits for a protocol in one go")
    p_all.add_argument("--chb-mit-dir", required=True)
    p_all.add_argument("--out", required=True)
    p_all.add_argument("--protocol", type=int, default=3, choices=[1, 2, 3])
    p_all.add_argument("--window-seconds", type=float, default=1.0)
    p_all.add_argument("--include-windows", action="store_true")
    p_all.add_argument("--splits", default=str(Path(__file__).resolve().parents[1] / "spec" / "splits.json"))
    p_all.set_defaults(func=cmd_prepare_all)

    # -- score --
    p_score = sub.add_parser("score", help="Compute metrics from a predictions .npy against labels")
    p_score.add_argument("--predictions", required=True, help="Path to .npy with predicted class ints [N]")
    p_score.add_argument("--labels", required=True, help="Path to labels.npy from prepare step")
    p_score.add_argument("--event-ids", help="Optional: event_ids.npy for event-level metrics")
    p_score.add_argument("--window-times", help="Optional: window_times.npy for lead-time metric")
    p_score.add_argument("--event-onset-times", help="Optional: event_onset_times.json for lead-time metric")
    p_score.add_argument("--out", required=True, help="Output submission JSON path")
    p_score.add_argument("--model-name", required=True)
    p_score.add_argument("--model-description", default="")
    p_score.add_argument("--model-params", type=int, default=0)
    p_score.add_argument("--training-time-hours", type=float, default=0.0)
    p_score.add_argument("--hardware", default="")
    p_score.add_argument("--protocol", default="cross_patient_fixed")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
