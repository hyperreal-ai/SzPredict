# SzPredict

**A standardised benchmark for EEG seizure prediction on the CHB-MIT Scalp EEG Database.**

Seizure prediction research has a reproducibility problem. Published results use inconsistent evaluation protocols, varying preictal windows, and patient-specific training that inflates reported metrics. Cross-patient generalisation — the clinically relevant benchmark — is rarely evaluated.

SzPredict fixes this. Fixed data splits. Defined evaluation protocols. Consistent metrics. Honest baselines.

> **Website:** [hyperreal.com.au/szpredict](https://hyperreal.com.au/szpredict/)
> **Organisation:** [HyperReal](https://hyperreal.com.au/) — Adelaide, Australia

---

## Why This Exists

Most published seizure prediction results report **patient-specific** accuracy: train on patient X, test on patient X. These numbers look impressive (95%+) but say nothing about whether a model can predict seizures in a new, unseen patient — which is the only scenario that matters clinically.

SzPredict provides three evaluation protocols of increasing difficulty, so methods can be compared on equal footing:

| Protocol | Method | Clinical Relevance |
|----------|--------|-------------------|
| **Protocol 1** | Patient-Specific | Literature comparison (inflated) |
| **Protocol 2** | Leave-One-Patient-Out | Gold standard generalisation |
| **Protocol 3** | Cross-Patient Fixed Split | Practical deployment evaluation |

**Protocol 3 is our primary benchmark.** One training run. Evaluate on patients the model has never seen. No cherry-picking.

---

## Dataset

**CHB-MIT Scalp EEG Database** ([PhysioNet](https://physionet.org/content/chbmit/1.0.0/))
- 24 paediatric subjects
- 844+ hours of continuous scalp EEG
- 198 annotated seizure events
- 256 Hz sampling rate, 23 channels (standard 10-20 montage)

### Label Definitions (Fixed)

| Label | Name | Definition |
|:---:|------|------------|
| 0 | **Interictal** | Normal EEG. Not within preictal or ictal window. |
| 1 | **Preictal** | 5-minute window (300s) immediately before seizure onset. |
| 2 | **Ictal** | During seizure (onset to offset, as annotated). |

**Preictal window = 300 seconds.** This is fixed. Published work varies this from 5 to 120 minutes — a major source of incomparability. We standardise at 5 minutes: the clinically actionable horizon (time to administer rescue medication or reach safety).

### Exclusion Zones

- **Post-ictal:** 5 minutes after seizure offset excluded (distinct EEG characteristics confound classification)
- **Consecutive seizures:** Inter-seizure periods under 10 minutes excluded entirely

---

## Evaluation Protocols

### Protocol 1 — Patient-Specific

Train and evaluate within each patient. Chronological 70/30 split (no shuffling — preserves temporal ordering). Per-patient metrics averaged across all patients.

*Included for literature comparison. Expected to overstate real-world performance.*

### Protocol 2 — Leave-One-Patient-Out (LOPO)

For each of 24 patients: train on the other 23, evaluate on the held-out patient. 24 full training runs. The gold standard for cross-patient generalisation.

*Computationally expensive. Expected 10–30% performance drop vs patient-specific.*

### Protocol 3 — Cross-Patient Fixed Split

| Set | Patients | Count |
|-----|----------|:---:|
| **Train** | chb01, chb03, chb05, chb06, chb08, chb10, chb12, chb13, chb15, chb18, chb20, chb24 | 12 |
| **Validation** | chb04, chb09, chb14, chb16, chb21 | 5 |
| **Test** | chb02, chb07, chb11, chb17, chb19, chb22, chb23 | 7 |

**Split rationale:**
- High-seizure patients (chb12, chb15, chb24) in train to maximise training signal
- Mix of seizure frequencies across all sets
- Deterministic — no randomness, fully reproducible
- Validation for hyperparameter tuning and early stopping only
- Test set metrics are the final reported numbers (never used for model selection)

---

## Metrics

### Primary

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Sensitivity** | TP / (TP + FN) for seizure classes | How many seizures we catch |
| **Specificity** | TN / (TN + FP) for interictal class | How often we correctly say "no seizure coming" |
| **Seizure Discrimination** | Correct / Total for preictal + ictal combined | Overall seizure recognition rate |
| **False Negative Rate** | Seizure windows misclassified as interictal / Total seizure windows | The dangerous errors — missed seizures |
| **Balanced Accuracy** | Mean of per-class accuracies | Robust to class imbalance |

### Secondary

| Metric | Definition |
|--------|------------|
| **F1 (macro)** | Harmonic mean of precision & recall, macro-averaged |
| **Confusion Matrix** | Full 3×3 error distribution (interictal / preictal / ictal) |
| **Miss Rate** | Seizure events with zero correct predictions / Total seizure events |
| **Detection Lead Time** | Time between first correct preictal prediction and seizure onset |

### Degenerate Detector

A model that predicts the same class for >95% of inputs is flagged as **degenerate**. This is common with aggressive class weighting on imbalanced datasets and should be reported honestly rather than concealed.

---

## Baseline Results

Four model generations evaluated across two protocols. These are our first-generation baselines — published to demonstrate the benchmark framework and establish honest starting points.

### Own-Validation Results

| Model | Seizure Disc. | Specificity | Balanced Acc. | Status |
|-------|:---:|:---:|:---:|--------|
| Gen 1 — Baseline | 0.0% | 3.3% | 1.1% | Degenerate |
| Gen 2 — Weighted Loss | 12.7% | 0.0% | 10.2% | Partial |
| Gen 3 — Enhanced Tokenizer | 80.6% | 0.0% | 33.2% | Degenerate |
| Gen 3b — LR Tuned | 80.8% | 0.2% | 33.4% | Degenerate |

### Protocol 3 — Cross-Patient Results

| Model | Seizure Disc. | Specificity | Balanced Acc. | Status |
|-------|:---:|:---:|:---:|--------|
| Gen 3 — Enhanced Tokenizer | 88.7% | 0.0% | 33.2% | Degenerate |
| Gen 3b — LR Tuned | 88.5% | 0.5% | 33.3% | Degenerate |

*33.3% balanced accuracy = random chance for a 3-class problem.*

### Key Finding — Degenerate Classification

All baseline models exhibit degenerate behaviour under rigorous evaluation. Models trained with aggressive class weighting (75× preictal, 300× ictal) learn to predict the highest-weighted class for nearly all inputs — achieving high seizure discrimination scores while completely failing to distinguish between brain states.

**The benchmark caught what training evaluation missed.** This is exactly why SzPredict exists.

Our next generation addresses this directly: equal class weights with balanced accuracy model selection instead of validation loss. Results forthcoming.

### Hardware

- **Training:** Dell C4130 (4× Pascal P100 16GB). No cloud compute.
- **Benchmarking:** Surface Pro 5 (Intel i7-7660U, no GPU). Full Protocol 3 evaluation (~23,000 batches) runs in ~48 hours on CPU.

---

## Submission Format

Results are submitted as JSON files conforming to the schema below. See [`results/`](results/) for examples from our baseline runs.

```json
{
  "benchmark_version": "0.1",
  "protocol": "cross_patient_fixed",
  "model_name": "Your-Model-Name",
  "model_description": "Brief architecture description",
  "model_params": 28000000,
  "training_time_hours": null,
  "hardware": "GPU type and count",

  "per_class": {
    "interictal": {"correct": 0, "total": 0, "accuracy": 0.0},
    "preictal":   {"correct": 0, "total": 0, "accuracy": 0.0},
    "ictal":      {"correct": 0, "total": 0, "accuracy": 0.0}
  },
  "confusion_matrix": {
    "interictal": {"pred_interictal": 0, "pred_preictal": 0, "pred_ictal": 0},
    "preictal":   {"pred_interictal": 0, "pred_preictal": 0, "pred_ictal": 0},
    "ictal":      {"pred_interictal": 0, "pred_preictal": 0, "pred_ictal": 0}
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
6. **Report all metrics** — cherry-picking is not permitted
7. **Report training details** — architecture, parameters, training time, hardware
8. **Flag degenerate models** — if >95% of predictions are a single class, report it

---

## Comparison with Existing Benchmarks

| Benchmark | Task | Dataset | Cross-Patient | Open Source |
|-----------|------|---------|:---:|:---:|
| SzCORE | Seizure *detection* | Multiple | Yes | Yes |
| **SzPredict** | Seizure *prediction* | CHB-MIT | **Yes** | **Yes** |
| Most published work | Prediction | CHB-MIT | No | No |

**SzCORE** benchmarks seizure *detection* (identifying seizures as they happen). **SzPredict** targets seizure *prediction* (forecasting before onset). These are complementary but distinct clinical problems. Detection helps during a seizure. Prediction helps *prevent* one.

---

## Research Roadmap

SzPredict is part of a broader research programme in temporal signal intelligence:

1. **Seizure Prediction** *(current)* — Cross-patient generalisation on CHB-MIT
2. **EEG-to-fMRI Super-Resolution** *(next)* — Predict spatial brain maps from temporal EEG data
3. **Brain-Computer Interface** *(future)* — Decode intention from non-invasive neural signals

The core thesis: temporal signals with multi-scale structure benefit from wavelet decomposition before sequence modelling. This applies across domains — the same architectural approach that predicts seizures also powers our [financial market prediction](https://hyperreal.com.au/finform/) system (1,100+ live paper trades).

---

## Citation

If you use SzPredict in your research, please cite:

```
@misc{szpredict2026,
  title={SzPredict: A Standardised Benchmark for EEG Seizure Prediction},
  author={HyperReal},
  year={2026},
  url={https://github.com/hyperreal-ai/SzPredict}
}
```

---

## Contact

- **Benchmark enquiries:** [szpredict@hyperreal.com.au](mailto:szpredict@hyperreal.com.au)
- **Website:** [hyperreal.com.au/szpredict](https://hyperreal.com.au/szpredict/)

---

## License

Benchmark specification, evaluation protocols, and data splits: **MIT License**

Baseline model architectures and training code: Proprietary (HyperReal). The benchmark is open — the models that compete on it need not be.

---

*Built in Adelaide, Australia. Trained on a Dell C4130. Benchmarked on a Surface Pro 5. No cloud compute. No venture funding. Just work that matters.*
