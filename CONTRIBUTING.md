# Contributing to SzPredict

The benchmark only becomes valuable when others run their own models against it. We welcome submissions.

## What we accept

- **Model results** on any of Protocols 1–4, submitted as JSON to `results/`
- **Documentation fixes** and README improvements
- **Tooling improvements** — better mock data generators, additional visualizations, metric extensions
- **Bug reports** in the metrics module or baselines

## What we don't accept (yet)

- Changes to protocol definitions or dataset splits — those are stable by design. If you believe a protocol should change, open an issue for discussion before a PR.
- Third-party model code that violates its upstream license.

## Submitting a model result

1. **Train** your model targeting the protocol of your choice (typically Protocol 3).
2. **Evaluate** using the consistent metric suite:
   ```python
   from szpredict.metrics import compute_all

   report = compute_all(predictions, labels, event_ids=event_ids)  # optional: window_times, event_onset_times
   ```
3. **Write a submission JSON** to `results/<model_name>_<protocol>.json`:
   ```json
   {
     "benchmark_version": "0.1",
     "protocol": "cross_patient_fixed",
     "model_name": "MyModel-v1",
     "model_description": "2-layer LSTM with attention head. Trained on raw EEG at 256 Hz.",
     "model_params": 1200000,
     "training_time_hours": 4.2,
     "hardware": "1x RTX 4090",
     "preprocessing": "Bandpass 0.5-45 Hz, z-score per channel",
     "post_processing": "5s sliding window majority vote (k=8 of 10)",
     "notes": "Anything a reviewer would want to know.",

     "per_class": { ... },
     "confusion_matrix": { ... },
     "sensitivity": 0.0,
     "specificity": 0.0,
     "balanced_accuracy": 0.0,
     "seizure_discrimination": 0.0,
     "f1_macro": 0.0,
     "false_positive_rate": 0.0,
     "degenerate": {"is_degenerate": false, "single_class": null, "threshold": 0.95},
     "clinical_utility": { ... },
     "miss_rate": { ... },
     "lead_time": { ... }
   }
   ```
   The `compute_all()` function already produces most fields; you add the descriptive metadata (`model_name`, `model_description`, etc.).

4. **Open a Pull Request** against `main`. Include in the PR description:
   - Link to model code or paper (if applicable; proprietary models OK — just describe architecture in prose)
   - Training pipeline summary (preprocessing, data augmentation, loss, optimiser)
   - Post-processing applied (smoothing, thresholding, refractory periods, calibration)
   - Hardware and training duration
   - Any deviation from the protocol spec

5. **Degenerate results are welcome.** If your model collapses to a single class, report it honestly — those results are genuinely useful for the field. The `is_degenerate` flag surfaces the failure mode instead of hiding it.

## Reproducibility requirements

Submissions that can't be independently reproduced won't be merged, but we're lenient about HOW reproducibility is achieved:

- **Ideal:** submit model weights + inference script. Reviewers can re-run.
- **Good:** submit training script + seed + hyperparameters. Reviewers can re-train.
- **Acceptable:** architecture description + hyperparameters detailed enough that a competent engineer can replicate.

Proprietary models are explicitly welcome (architecture description can be high-level) as long as the methodology is documented enough to interpret the numbers.

## What reviewers check

When you open a PR, maintainers will check:
- [ ] Submission JSON validates against the schema
- [ ] Fields present: `benchmark_version`, `protocol`, `model_name`, `model_description`, all required metrics
- [ ] Protocol spec adhered to (correct preictal window, split, etc.)
- [ ] Post-processing disclosed
- [ ] Degeneracy honestly flagged if applicable
- [ ] FPR/h is realistic (balanced-subset FAR inflation is a known trap — see Pale et al. 2023)
- [ ] No obvious test-set leakage

## Questions / clarifications

- Open a GitHub issue for protocol questions
- Email [szpredict@hyperreal.com.au](mailto:szpredict@hyperreal.com.au) for anything else

## Code of conduct

Be respectful. Seizure prediction research matters to real patients and families. Keep discussions technical and constructive.
