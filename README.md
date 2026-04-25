# SzPredict

**An open benchmark for EEG seizure prediction on the CHB-MIT Scalp EEG Database.**

Standardised evaluation protocols, consistent metrics, and a cross-patient testbed that surfaces failure modes the field has been hiding.

> **Website:** [hyperreal.com.au/szpredict](https://hyperreal.com.au/szpredict/)
> **Organisation:** [HyperReal](https://hyperreal.com.au/) — Adelaide, Australia
> **License:** MIT (benchmark + baselines). Individual model architectures may have their own licensing.

---

## Quickstart

Five minutes from `git clone` to running baselines.

```bash
git clone https://github.com/hyperreal-ai/SzPredict.git
cd SzPredict
./install.sh                              # Python deps + CHB-MIT setup (interactive)

### install.sh repares the environment (venv), runs a mock pipeline test, downloads CHB-MIT data (or accepts location of dataset if already downloaded), prepares the data, trains a CNN model, tests that CNN model, benchmarks it and produces a protocol-3 result.

# Generate mock labels+windows for a quick pipeline test (no CHB-MIT needed)
python scripts/make_mock_labels.py --out data/mock --n 10000 --events 15 --with-windows

# Run the three reference baselines
python -m baselines.baseline_random   --labels data/mock/labels.npy --event-ids data/mock/event_ids.npy --out results/mock_random.json
python -m baselines.baseline_majority --labels data/mock/labels.npy --event-ids data/mock/event_ids.npy --out results/mock_majority.json
python -m baselines.baseline_cnn train --train-x data/mock/windows.npy --train-y data/mock/labels.npy \
                                        --val-x   data/mock/windows.npy --val-y   data/mock/labels.npy \
                                        --out runs/cnn_mock --epochs 15
python -m baselines.baseline_cnn eval  --ckpt runs/cnn_mock/best.pt \
                                        --test-x  data/mock/windows.npy --test-y  data/mock/labels.npy \
                                        --test-events data/mock/event_ids.npy \
                                        --out results/mock_cnn.json

# View metrics
python -c "import json; print(json.dumps(json.load(open('results/mock_cnn.json')), indent=2))"
```

Real CHB-MIT pipeline follows the same pattern once `./install.sh` has set up the dataset — just swap the `--labels` / `--windows` paths.

**Got a spare weekend and a GPU?** Swap the `baseline_cnn.py` backbone for your own architecture and run it through the same benchmark. Submit results via PR to `results/`. See [Contributing](#contributing).

---

## Why This Exists

Most published seizure prediction results report **patient-specific** accuracy: train on patient X, test on patient X. These numbers look impressive (95%+) but say nothing about whether a model can predict seizures in a new, unseen patient — which is the scenario that matters clinically.

Across a 19-paper review of recent CHB-MIT work, reported sensitivities range from **58% to nearly 100%** — not because models differ that much, but because papers use incompatible task conventions, preictal windows, patient cohorts, and post-processing rules. Numbers look impressive in isolation; most aren't directly comparable.

SzPredict pins every axis. See [`lit_review/corpus_synthesis.md`](lit_review/corpus_synthesis.md) for the full methodological analysis.

---

## Evaluation Protocols

| Protocol | Method | Clinical Relevance |
|----------|--------|-------------------|
| **Protocol 1** | Patient-Specific | Literature comparison (inflated) |
| **Protocol 2** | Leave-One-Patient-Out (LOPO) | Gold-standard generalisation |
| **Protocol 3** | Cross-Patient Fixed Split | **Primary benchmark** — practical deployment evaluation |
| **Protocol 4** | Transition Timing | Clinical utility — how early before onset does the model warn? |

**Protocol 4 is the metric that actually matters.** Not 'classification accuracy on balanced test segments' but 'how many minutes before seizure onset does the model reliably warn?' The clinical target is specific: a device that knows a patient's interictal baseline and says *"Caution — Preictal detected"* or, at full strength, *"Seizure Predicted — onset in approximately 13 minutes."*

See [`spec/BENCHMARK_SPEC.md`](spec/BENCHMARK_SPEC.md) for full protocol definitions and [`spec/splits.json`](spec/splits.json) for Protocol 3's fixed patient split.

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

**Preictal window = 300 seconds.** Fixed. Published work varies this from 5 to 120 minutes — a major source of incomparability. We standardise at 5 minutes: the clinically actionable horizon (time to administer rescue medication or reach safety).

**Exclusion zones:**
- Post-ictal: 5 minutes after seizure offset excluded
- Consecutive seizures: inter-seizure periods under 10 minutes excluded entirely

---

## Installation

```bash
./install.sh
```

Interactive installer handles CHB-MIT dataset setup:
1. **Download fresh** from PhysioNet (~42 GB, 20–60 min depending on bandwidth)
2. **Use existing local copy** (provide path; installer symlinks and verifies structure)
3. **Skip dataset** (install code only; use mock data for pipeline testing)

Dependencies: Python 3.8+, numpy, torch. Full list in `requirements.txt`.

---

## Metrics

Every submission is evaluated on a consistent suite. Primary metrics:

- **Sensitivity** — TP / (TP + FN) on seizure classes
- **Specificity** — TN / (TN + FP) on interictal
- **Balanced Accuracy** — mean of per-class recalls (robust to class imbalance)
- **F1 (macro)**
- **False Positive Rate per hour** (FPR/h)
- **Seizure Discrimination** — (preictal + ictal correct) / (preictal + ictal total). Preictal predicted as ictal counts as 'correct' — both are seizure states. *Not* equivalent to sensitivity.

Protocol 4 additional metrics:
- **Preictal Lead Time** — minutes before onset the model first correctly flags preictal
- **Transition Detection Rate** — fraction of seizure events detected in advance

**Degenerate Detector:** a model predicting the same class for >95% of inputs is flagged as **degenerate** in its submission. Common failure mode under aggressive class weighting. Report honestly; the benchmark exists to surface it.

---

## Reference Baselines

Three deliberately-minimal baselines in `baselines/`:

| Baseline | Purpose | Script |
|---|---|---|
| **Random** | Uniform-random 3-class predictor. Pure floor. | `baselines/baseline_random.py` |
| **Majority-Interictal** | Always predicts interictal. Shows the degenerate trap. | `baselines/baseline_majority.py` |
| **Simple 1D-CNN** | Minimal CNN on raw EEG windows. Equal class weights, balanced-accuracy model selection — swap in your own backbone. | `baselines/baseline_cnn.py` |

All three use only standard PyTorch components. No proprietary architectures.

---

## Submission Format

Submit a JSON conforming to the schema documented in `spec/BENCHMARK_SPEC.md`. See `results/` for examples.

```json
{
  "benchmark_version": "0.1",
  "protocol": "cross_patient_fixed",
  "model_name": "Your-Model-Name",
  "model_description": "Brief architecture description",
  "model_params": 28000000,
  "training_time_hours": 12.5,
  "hardware": "1x RTX 5090",
  "per_class": { ... },
  "confusion_matrix": { ... },
  "sensitivity": 0.0,
  "specificity": 0.0,
  "balanced_accuracy": 0.0,
  "seizure_discrimination": 0.0,
  "f1_macro": 0.0,
  "degenerate": { "is_degenerate": false },
  "clinical_utility": { ... },
  "miss_rate": { ... },
  "lead_time": { ... }
}
```

Generate this automatically with `szpredict.metrics.compute_all(predictions, labels, event_ids)`.

---

## Contributing

We welcome community submissions. To add a result:

1. Train your model under the protocol you're targeting (usually Protocol 3).
2. Evaluate on the test split using `szpredict.metrics.compute_all()` on your predictions.
3. Save the submission JSON to `results/<your_model_name>_<protocol>.json`.
4. Open a PR. Include:
   - Model architecture description (or paper link)
   - Training data + preprocessing pipeline
   - Hardware used + training time
   - Any post-processing applied (smoothing, thresholding, etc.)
   - Reproducibility checklist (see `spec/BENCHMARK_SPEC.md`)

All reasonable submissions accepted. Degenerate results are **welcome** — publishing negative results is part of the benchmark's mission.

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

## Methodological Context

Behind this benchmark sits a careful review of the CHB-MIT seizure-prediction literature:

- [`lit_review/corpus_synthesis.md`](lit_review/corpus_synthesis.md) — 19-paper synthesis. 4 task conventions, 10+ rosetta dimensions, 6-group methodological convergence, 5 scarcity-mitigation strategies, 8-role corpus map.
- [`lit_review/phase1_triage.md`](lit_review/phase1_triage.md) — Paper-by-paper triage with applicability scoring.

The short version: the field has been comparing apples to oranges. SzPredict pins every axis so your result and theirs become directly comparable for the first time.

**Key external reference:** Pale, Teijeiro & Atienza (2023), *Importance of methodological choices in data manipulation for validating epileptic seizure detection models* (arXiv:2302.10672). They called out the same problems this benchmark addresses. Cite them early and often.

---

## Research Roadmap

SzPredict is part of a broader research programme in temporal signal intelligence:

1. **Seizure Prediction** *(current)* — Cross-patient generalisation + clinical transition timing on CHB-MIT
2. **EEG-to-fMRI Super-Resolution** *(next)* — Predict spatial brain maps from temporal EEG
3. **Brain-Computer Interface** *(future)* — Decode intention from non-invasive neural signals

Core thesis: temporal signals with multi-scale structure benefit from explicit wavelet decomposition before sequence modelling. This applies across domains — the architectural approach that predicts seizures also powers our [financial market prediction](https://hyperreal.com.au/finform/) system.

---

## Citation

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

## Comparison with Existing Benchmarks

| Benchmark | Task | Dataset | Cross-Patient | Open Source |
|-----------|------|---------|:---:|:---:|
| SzCORE | Seizure *detection* | Multiple | Yes | Yes |
| **SzPredict** | Seizure *prediction + transition timing* | CHB-MIT | **Yes** | **Yes** |
| Most published work | Prediction | CHB-MIT | No | No |

SzCORE benchmarks seizure *detection* (identifying seizures as they happen). SzPredict targets seizure *prediction* (forecasting before onset) and *clinical transition timing* (how early the warning fires). Complementary but distinct clinical problems. Detection helps during a seizure. Prediction helps *prevent* one.

---

*Built in Adelaide, Australia. Trained across progressively constrained hardware (Dell C4130 4×P100 → Surface Pro 5 CPU → Ryzen 5900X + A100 32GB) — under $2,000 total, no cloud compute. The constraints forced architectural choices toward efficient inference, which happens to be the same constraint wearable seizure-warning hardware requires.*
