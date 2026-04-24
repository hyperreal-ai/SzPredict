# SzPredict Benchmark Specification v0.1

## Standardised Evaluation Protocol for EEG Seizure Prediction on CHB-MIT

---

## Overview

This benchmark defines a standardised evaluation protocol for seizure **prediction** (not detection) on the CHB-MIT Scalp EEG Database. It addresses a critical gap: most published results use patient-specific evaluation with varying methodology, making cross-paper comparison meaningless. This benchmark enforces consistent data splits, label definitions, metrics, and evaluation protocols.

**Key differentiator from SzCORE:** SzCORE benchmarks seizure *detection* (identifying seizures as they happen). This benchmark targets seizure *prediction* (forecasting before onset) — a complementary but distinct clinical problem.

---

## Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet)
- 24 patients (chb01–chb24), paediatric subjects
- 844 hours of continuous EEG
- 198 seizure events annotated with onset/offset times
- 256 Hz sampling rate, 23 channels (standard 10-20 system)
- Freely available: https://physionet.org/content/chbmit/1.0.0/

### Patient Summary

| Patient | Seizure Files | Total Seizures | Notes |
|---------|:---:|:---:|-------|
| chb01 | 7 | 7 | |
| chb02 | 3 | 3 | |
| chb03 | 7 | 7 | |
| chb04 | 3 | 4 | |
| chb05 | 5 | 5 | |
| chb06 | 7 | 10 | |
| chb07 | 3 | 3 | |
| chb08 | 5 | 5 | |
| chb09 | 3 | 4 | |
| chb10 | 7 | 7 | |
| chb11 | 3 | 3 | |
| chb12 | 12 | 40 | High seizure count |
| chb13 | 8 | 12 | |
| chb14 | 7 | 8 | |
| chb15 | 14 | 20 | High seizure count |
| chb16 | 6 | 10 | |
| chb17 | 3 | 3 | |
| chb18 | 6 | 6 | |
| chb19 | 3 | 3 | |
| chb20 | 6 | 8 | |
| chb21 | 4 | 4 | |
| chb22 | 3 | 3 | |
| chb23 | 3 | 7 | |
| chb24 | 12 | 16 | High seizure count |

---

## Label Definitions (Fixed)

All protocols use the same label definitions:

| Label | Name | Definition |
|:---:|------|------------|
| 0 | **Interictal** | Normal EEG. Not within the preictal or ictal window. |
| 1 | **Preictal** | The **5-minute window** (300 seconds) immediately before seizure onset. |
| 2 | **Ictal** | During the seizure (onset to offset, as annotated). |

**Preictal window = 300 seconds.** This is fixed across all protocols. Published work varies this from 5 to 120 minutes — a major source of incomparability. We standardise at 5 minutes as the clinically actionable prediction horizon (enough time to administer rescue medication or seek safety).

### Exclusion Zones

- **Post-ictal:** The 5-minute window after seizure offset is excluded (neither interictal nor preictal). Post-ictal EEG has distinct characteristics that would confound interictal classification.
- **Consecutive seizures:** If two seizures are within 10 minutes of each other, the inter-seizure period is excluded entirely.

---

## Evaluation Protocols

### Protocol 1: Patient-Specific (Within-Patient)

**Purpose:** Direct comparison with published literature (90%+ of papers use this).

**Method:**
- For each patient independently:
  - Chronological split: first 70% of recordings = train, last 30% = test
  - No shuffling — preserves temporal ordering (critical for non-stationarity)
  - Train and evaluate a model per patient
- Report per-patient metrics, then average across patients

**Reporting:**
- Per-patient scores (table)
- Mean ± std across patients
- Median across patients (robust to outliers)

### Protocol 2: Leave-One-Patient-Out (LOPO)

**Purpose:** Gold standard for cross-patient generalisation. Tests whether a model can predict seizures in a completely unseen patient.

**Method:**
- For each of the 24 patients:
  - Train on the other 23 patients
  - Test on the held-out patient
  - Record metrics for the held-out patient
- Report per-patient metrics and averages

**Notes:**
- This is computationally expensive (24 full training runs)
- Expected performance drop of 10-30% vs patient-specific
- This is the protocol that separates genuine generalisation from overfitting

**Reporting:**
- Per-patient scores when held out (table)
- Mean ± std across held-out patients
- Which patients are hardest/easiest to generalise to

### Protocol 3: Cross-Patient (Fixed Split)

**Purpose:** Practical deployment evaluation. Train once, evaluate on unseen patients.

**Standardised Split:**

| Set | Patients | Purpose |
|-----|----------|---------|
| **Train** | chb01, chb03, chb05, chb06, chb08, chb10, chb12, chb13, chb15, chb18, chb20, chb24 | 12 patients, ~120 seizure events |
| **Val** | chb04, chb09, chb14, chb16, chb21 | 5 patients, hyperparameter tuning |
| **Test** | chb02, chb07, chb11, chb17, chb19, chb22, chb23 | 7 patients, final evaluation |

**Split rationale:**
- 50/21/29% split by patients (12/5/7)
- High-seizure patients (chb12, chb15, chb24) in train to maximise training signal
- Mix of seizure frequencies across all sets
- chb02 in test (not val) to prevent val-driven overfitting to a single patient
- Deterministic — no randomness, fully reproducible

**Notes:**
- Single training run
- Val set for hyperparameter tuning and early stopping
- Test set metrics are the final reported numbers (never used for model selection)

---

## Metrics

All protocols report the same metrics:

### Primary Metrics

| Metric | Definition | Clinical Relevance |
|--------|------------|-------------------|
| **Sensitivity (Recall)** | TP / (TP + FN) for seizure classes | How many seizures we catch |
| **Specificity** | TN / (TN + FP) for interictal class | How often we correctly say "no seizure" |
| **Seizure Discrimination** | Correct / Total for preictal + ictal combined | Overall seizure recognition rate |
| **False Negative Rate** | Seizure windows misclassified as interictal / Total seizure windows | The dangerous errors — missed seizures |

### Secondary Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| **Preictal Accuracy** | Correct preictal / Total preictal | Prediction-specific accuracy |
| **Ictal Accuracy** | Correct ictal / Total ictal | Detection accuracy (during seizure) |
| **F1 (macro)** | Harmonic mean of precision & recall, macro-averaged | Balances precision and recall across classes |
| **Confusion Matrix** | 3x3 matrix (interictal/preictal/ictal) | Full error distribution |

### Clinical Safety Metric

| Metric | Definition | Target |
|--------|------------|--------|
| **Miss Rate** | Seizure events with ZERO preictal or ictal predictions / Total seizure events | 0% (no completely missed seizures) |

A seizure "event" is a continuous ictal period. If ANY window in the preictal or ictal phase is correctly classified, the event is "detected." The miss rate measures complete failures — seizures with no warning at all.

---

## Evaluation Procedure

### Window-Level Evaluation (Default)

1. Extract all label positions from the evaluation data
2. For each label position, extract a context window (seq_len tokens ending at the label)
3. Run the model on each context window
4. Compare predicted label to ground truth
5. Compute metrics

### Event-Level Evaluation (Supplementary)

1. Group consecutive preictal/ictal windows into seizure events
2. A seizure event is "predicted" if ≥1 preictal window is correctly classified
3. Report event-level sensitivity and lead time

### Detection Lead Time (Supplementary)

For correctly predicted seizure events:
- Record the time (in seconds) between the first correct preictal prediction and seizure onset
- Report: mean, median, min, max lead time
- Maximum possible: 300 seconds (full preictal window)

---

## Submission Format

Results are submitted as a JSON file:

```json
{
  "benchmark_version": "0.1",
  "protocol": "cross_patient_fixed",
  "model_name": "NeuroWave-v1",
  "model_description": "Modified transformer 384d/8L with wavelet-based tokenization",
  "model_params": 28000000,
  "training_time_hours": null,
  "hardware": "NVIDIA A100 32GB",

  "per_class": {
    "interictal": {"correct": 0, "total": 0, "accuracy": 0.0},
    "preictal": {"correct": 0, "total": 0, "accuracy": 0.0},
    "ictal": {"correct": 0, "total": 0, "accuracy": 0.0}
  },
  "confusion_matrix": {
    "interictal": {"pred_interictal": 0, "pred_preictal": 0, "pred_ictal": 0},
    "preictal": {"pred_interictal": 0, "pred_preictal": 0, "pred_ictal": 0},
    "ictal": {"pred_interictal": 0, "pred_preictal": 0, "pred_ictal": 0}
  },
  "seizure_discrimination": 0.0,
  "false_negative_rate": 0.0,
  "sensitivity": 0.0,
  "specificity": 0.0,
  "f1_macro": 0.0,
  "miss_rate": 0.0,
  "lead_time": {
    "mean_seconds": 0.0,
    "median_seconds": 0.0,
    "min_seconds": 0.0,
    "max_seconds": 0.0
  },

  "per_patient": {
    "chb02": {"sensitivity": 0.0, "specificity": 0.0, "seizure_discrimination": 0.0}
  }
}
```

---

## Reproducibility Requirements

1. **Fixed random seeds** for any stochastic operations
2. **No test set leakage** — test set never used for model selection or hyperparameter tuning
3. **Chronological ordering** preserved within each patient
4. **Preictal window fixed at 300 seconds** — no variation
5. **Post-ictal exclusion zone: 300 seconds** — no variation
6. **Report all metrics** — cherry-picking metrics is not permitted
7. **Report training details** — architecture, params, training time, hardware

---

## Changelog

- **v0.1** (2026-02-28): Initial specification. Three evaluation protocols, standardised metrics, CHB-MIT data splits defined.
