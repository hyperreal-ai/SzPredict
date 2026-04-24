"""
SzPredict metrics module
========================
Compute the full benchmark metric suite from per-window predictions and labels.

Window labels (fixed across all protocols):
    0 = interictal
    1 = preictal   (300s window before seizure onset)
    2 = ictal      (during seizure)

Inputs:
    predictions: np.ndarray of int, shape [N]
    labels:      np.ndarray of int, shape [N]
    event_ids:   optional np.ndarray of int, shape [N] — seizure event each window belongs to
                 (-1 or any negative for interictal windows). Needed for event-level metrics.
    window_times: optional np.ndarray of float, shape [N] — seconds from some reference.
                  Needed for lead-time metric (per-event first-correct-prediction timing).

All scalar metrics are returned as native floats (not numpy). Safe to JSON-serialize.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

CLASS_NAMES = ["interictal", "preictal", "ictal"]
N_CLASSES = 3
SEIZURE_CLASSES = (1, 2)  # preictal + ictal
DEGENERATE_THRESHOLD = 0.95


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return 3x3 confusion matrix. rows = true, cols = predicted."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < N_CLASSES and 0 <= p < N_CLASSES:
            cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> dict:
    """Per-class recall (TP / (TP+FN)) — what fraction of each class did we catch?"""
    out = {}
    for i, name in enumerate(CLASS_NAMES):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        out[name] = {
            "correct": correct,
            "total": total,
            "accuracy": _safe_div(correct, total),
        }
    return out


def balanced_accuracy(cm: np.ndarray) -> float:
    """Mean per-class recall. Robust to class imbalance."""
    recalls = []
    for i in range(N_CLASSES):
        total = cm[i].sum()
        if total > 0:
            recalls.append(cm[i, i] / total)
    return float(np.mean(recalls)) if recalls else 0.0


def seizure_discrimination(cm: np.ndarray) -> float:
    """(preictal+ictal correct) / (preictal+ictal total).
    Not equivalent to sensitivity — counts preictal-predicted-as-ictal as 'correct'
    in the sense that both are seizure classes. Matches our README definition."""
    numer = int(cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2])
    denom = int(cm[1].sum() + cm[2].sum())
    return _safe_div(numer, denom)


def sensitivity(cm: np.ndarray) -> float:
    """Recall for seizure classes (preictal + ictal) vs interictal.
    Binary-collapsed: did we say 'seizure happening or about to' when it was?"""
    # True seizure → predicted seizure (any of preictal/ictal)
    tp = int(cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2])
    # True seizure → predicted interictal
    fn = int(cm[1, 0] + cm[2, 0])
    return _safe_div(tp, tp + fn)


def specificity(cm: np.ndarray) -> float:
    """Recall for interictal class. Did we say 'no seizure' when there wasn't?"""
    tn = int(cm[0, 0])
    fp = int(cm[0, 1] + cm[0, 2])
    return _safe_div(tn, tn + fp)


def false_negative_rate(cm: np.ndarray) -> float:
    """Fraction of seizure windows classified as interictal. The dangerous error."""
    fn = int(cm[1, 0] + cm[2, 0])
    total_seizure = int(cm[1].sum() + cm[2].sum())
    return _safe_div(fn, total_seizure)


def false_positive_rate(cm: np.ndarray) -> float:
    """Fraction of interictal windows flagged as seizure. The 'false alarm' cost."""
    fp = int(cm[0, 1] + cm[0, 2])
    total_interictal = int(cm[0].sum())
    return _safe_div(fp, total_interictal)


def f1_macro(cm: np.ndarray) -> float:
    """Macro-averaged F1. Treats each class equally."""
    f1s = []
    for i in range(N_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def is_degenerate(preds: np.ndarray, threshold: float = DEGENERATE_THRESHOLD) -> tuple[bool, Optional[str]]:
    """Flag if model predicts one class for >95% of inputs."""
    if len(preds) == 0:
        return False, None
    for i, name in enumerate(CLASS_NAMES):
        frac = float((preds == i).sum()) / len(preds)
        if frac >= threshold:
            return True, name
    return False, None


# =============================================================================
# Clinical-utility metrics (Protocol 4 — real-world deployment framing)
# =============================================================================

def interictal_specificity_at(cm: np.ndarray, target_fraction: float = 0.99) -> float:
    """Specificity achieved on interictal class. Separate function for emphasis —
    this is the 'you're OK' certainty metric most relevant for clinical deployment.
    Returns the spec that was achieved (not a threshold-tuned answer)."""
    return specificity(cm)


def warning_rate_vs_false_alarm(cm: np.ndarray) -> dict:
    """Clinical-utility view: warnings-per-real-event vs false-alarms-per-interictal.
    Tracks TWO quantities a user experiences:
      - When a seizure is coming: how often does the device warn (sensitivity)
      - When NOT: how often does it falsely warn (false positive rate)
    This pair is the honest 'what does wearing this device feel like' view."""
    return {
        "warn_rate_given_seizure": sensitivity(cm),       # TP / (TP+FN)
        "false_alarm_rate": false_positive_rate(cm),      # FP / (FP+TN)
    }


# =============================================================================
# Event-level metrics (per-seizure, not per-window)
# =============================================================================

def miss_rate(preds: np.ndarray, labels: np.ndarray, event_ids: np.ndarray) -> dict:
    """Fraction of seizure EVENTS with zero correct preictal/ictal predictions.
    event_ids: negative for interictal, non-negative event identifier for seizure windows."""
    seizure_mask = (labels == 1) | (labels == 2)
    if not seizure_mask.any():
        return {"missed_events": 0, "total_events": 0, "miss_rate": 0.0}

    unique_events = np.unique(event_ids[seizure_mask])
    unique_events = unique_events[unique_events >= 0]  # exclude interictal sentinel
    total_events = len(unique_events)
    missed = 0
    for e in unique_events:
        mask = (event_ids == e) & seizure_mask
        # Event is "caught" if ANY of its preictal/ictal windows is predicted as preictal/ictal
        caught = bool(((preds[mask] == 1) | (preds[mask] == 2)).any())
        if not caught:
            missed += 1
    return {
        "missed_events": int(missed),
        "total_events": int(total_events),
        "miss_rate": _safe_div(missed, total_events),
    }


def detection_lead_time(
    preds: np.ndarray,
    labels: np.ndarray,
    event_ids: np.ndarray,
    window_times: np.ndarray,
    event_onset_times: dict,
) -> dict:
    """Lead time per event: seconds from first correct preictal prediction to onset.
    event_onset_times: {event_id: onset_time_seconds}.
    Reports mean/median/min/max across events."""
    lead_times = []
    seizure_mask = (labels == 1) | (labels == 2)
    unique_events = np.unique(event_ids[seizure_mask])
    unique_events = unique_events[unique_events >= 0]

    for e in unique_events:
        if e not in event_onset_times:
            continue
        mask = (event_ids == e) & (labels == 1)  # true preictal windows of this event
        if not mask.any():
            continue
        correct_preictal_mask = mask & (preds == 1)
        if not correct_preictal_mask.any():
            continue
        first_correct_time = window_times[correct_preictal_mask].min()
        onset = event_onset_times[e]
        lead = onset - first_correct_time
        if lead >= 0:
            lead_times.append(float(lead))

    if not lead_times:
        return {"mean_seconds": 0.0, "median_seconds": 0.0,
                "min_seconds": 0.0, "max_seconds": 0.0, "events_with_lead": 0}

    return {
        "mean_seconds": float(np.mean(lead_times)),
        "median_seconds": float(np.median(lead_times)),
        "min_seconds": float(np.min(lead_times)),
        "max_seconds": float(np.max(lead_times)),
        "events_with_lead": len(lead_times),
    }


# =============================================================================
# Full report (matches SzPredict submission schema)
# =============================================================================

def compute_all(
    predictions: np.ndarray,
    labels: np.ndarray,
    event_ids: Optional[np.ndarray] = None,
    window_times: Optional[np.ndarray] = None,
    event_onset_times: Optional[dict] = None,
) -> dict:
    """Compute every metric in one pass. Returns a dict that directly fills the
    submission JSON schema (see README.md submission format).

    event_ids / window_times / event_onset_times are optional — required only for
    event-level metrics (miss_rate, lead_time). If absent, those fields are null.
    """
    predictions = np.asarray(predictions, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    assert predictions.shape == labels.shape, \
        f"predictions {predictions.shape} != labels {labels.shape}"

    cm = confusion_matrix(predictions, labels)
    degen, degen_class = is_degenerate(predictions)

    report = {
        "n_windows": int(len(predictions)),
        "per_class": per_class_accuracy(cm),
        "confusion_matrix": {
            CLASS_NAMES[i]: {
                f"pred_{CLASS_NAMES[j]}": int(cm[i, j]) for j in range(N_CLASSES)
            }
            for i in range(N_CLASSES)
        },
        "seizure_discrimination": seizure_discrimination(cm),
        "sensitivity": sensitivity(cm),
        "specificity": specificity(cm),
        "false_negative_rate": false_negative_rate(cm),
        "false_positive_rate": false_positive_rate(cm),
        "f1_macro": f1_macro(cm),
        "balanced_accuracy": balanced_accuracy(cm),
        "degenerate": {
            "is_degenerate": degen,
            "single_class": degen_class,
            "threshold": DEGENERATE_THRESHOLD,
        },
        # Clinical-utility view (Protocol 4 framing)
        "clinical_utility": warning_rate_vs_false_alarm(cm),
    }

    if event_ids is not None:
        event_ids = np.asarray(event_ids, dtype=np.int64).ravel()
        assert event_ids.shape == predictions.shape
        report["miss_rate"] = miss_rate(predictions, labels, event_ids)

        if window_times is not None and event_onset_times is not None:
            window_times = np.asarray(window_times, dtype=np.float64).ravel()
            report["lead_time"] = detection_lead_time(
                predictions, labels, event_ids, window_times, event_onset_times
            )
        else:
            report["lead_time"] = None
    else:
        report["miss_rate"] = None
        report["lead_time"] = None

    return report
