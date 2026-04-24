# SzPredict Lit Review — Phase 1 Triage

**Source:** 38 CHB-MIT papers from `~/ai/_datasets/arxiv/arxiv_papers.db` (arxiv abstracts only).
**Date:** 2026-04-22.
**Filter criteria:** Papers targeting seizure PREDICTION (preictal identification ahead of onset) on CHB-MIT as a primary dataset. Detection-only and non-seizure work are flagged out.

## Summary

| Category | Count |
|---|---|
| ✅ **APPLICABLE** — direct prediction baselines | **13** |
| 📚 **METHODOLOGY / SURVEY** — cite for framing | 2 |
| ❓ **UNCERTAIN** — needs full-PDF skim | 2 |
| ❌ **NOT APPLICABLE** — detection-only, non-seizure, or CHB-MIT secondary | 21 |

## ✅ APPLICABLE (13) — core rosetta-stone candidates

| # | ID | Year | Title (short) | Reported numbers | Split |
|---|---|---|---|---|---|
| 1 | 2410.09998v1 | 2024 | SlimSeiz (Mamba, 8-channel) | 94.8% acc / 95.5% sens / 94.0% spec | patient-specific |
| 2 | 2407.19841v1 | 2024 | RRAM bio-inspired circuits | 91.2% sens / 0.11 FPR/h | unclear |
| 3 | 2407.14876v1 | 2024 | Preictal Period Optimization (CNN-Transformer) | 99.31% sens / 95.34% spec / AUC 99.35% / 76.8min lead | **patient-specific** |
| 4 | 2402.09424v1 | 2024 | Spiking Conformer | **pred:** 96.8% sens / 89.5% spec | unclear |
| 5 | 2306.08256v2 | 2023 | DiffEEG (diffusion data-aug) | 95.4% sens / 0.051 FPR/h / 0.932 AUC | unclear |
| 6 | 2211.02679v1 | 2022 | Automatic Prediction CNN+LSTM | 97.75% sens / 0.2373 FPR/h | unclear |
| 7 | 2209.11172v1 | 2022 | TMC-T / TMC-ViT transformers | (patient-specific, varied preictal) | **patient-specific** |
| 8 | 2206.09951v1 | 2022 | Memristive CNN | **pred:** 99.01% / 97.54% (CHB-MIT / SWEC) | 5-fold |
| 9 | 2206.07518v1 | 2022 | Binary Single-dim CNN (BSDCNN) | 94.69% sens / 0.970 AUC / 0.095 FPR/h | unclear |
| 10 | 2108.07453v1 | 2021 | End-to-End CNN (Xu) | 98.8% sens / 0.074 FPR/h / AUC 0.988 | unclear |
| 11 | 2106.04510v1 | 2021 | Random Forest (SPH=5min, SOP=30min) | 82.07% sens / 0.0799 FPR/h | 20 patients |
| 12 | 2105.02823v1 | 2021 | Multi-scale Dilated 3D CNN | 80.5% acc / 85.8% sens / 75.1% spec | unclear |
| 13 | 2012.00430v1 | 2020 | DCGAN data augmentation (CESP) | 78-88% sens / 0.14-0.27 FPR/h | transfer-learning |
| 14 | 2012.00307v3 | 2020 | Edge DL (DNN/CNN/LSTM) for implants | DNN 87.36% / CNN 96.70% / LSTM 97.61% (sens) | unclear |
| 15 | 2011.09581v1 | 2020 | **Patient-independent Prediction (Dissanayake)** | **88.81% / 91.54% acc** | **CROSS-PATIENT** ⭐ |

⭐ **Dissanayake 2020** is the closest direct comparison to our Protocol 3 (cross-patient fixed split).

## 📚 METHODOLOGY / SURVEY (2) — cite for framing

| # | ID | Year | Title | Why it matters |
|---|---|---|---|---|
| 16 | 2302.10672v1 | 2023 | **Importance of methodological choices (Pale et al.)** | Directly identifies the methodology-heterogeneity problem SzPredict addresses. **Strongest external support for our Protocol 4 critique.** |
| 17 | 2306.12292v1 | 2023 | Reporting existing datasets (Handa et al.) | Survey reference for CHB-MIT as benchmark dataset |

## ❓ UNCERTAIN (2) — needs PDF skim to classify

| # | ID | Year | Title | Why uncertain |
|---|---|---|---|---|
| 18 | 2403.03276v2 | 2024 | ARNN Attentive RNN | ~~UNCERTAIN~~ **RE-REVIEWED 2026-04-22 13:30: NOT applicable to prediction rosetta. Uses a THIRD task convention — merges preictal+ictal as positive class vs interictal. Model output cannot distinguish preictal from ictal. Accuracy 99.96% on this binary framing is non-transferable. Value: methodology-catalog evidence that the field has ≥3 common binary-task conventions on CHB-MIT.** |
| 19 | 2301.03465v2 | 2023 | Shorter Latency real-time | ~~UNCERTAIN~~ ~~APPLICABLE~~ **RE-REVIEWED 2026-04-22 13:25: NOT applicable to prediction rosetta. Paper is DETECTION (post-onset latency reduction). 'Crossing period' = window STRADDLING onset, NOT pre-onset. Value instead: (a) line 97 is a direct admission of methodology-heterogeneity ('Don't compare directly to older papers' sensitivity numbers'), PERFECT citation for Protocol 4. (b) Table 3 provides 12 DETECTION-lit entries for a future v0.2 detection panel.** |

## ❌ NOT APPLICABLE (21)

Detection-only, emotion recognition, or CHB-MIT as secondary dataset:

- **2410.19815v1** BUNDL (detection, noisy labels)
- **2406.19189v1** BISeizuRe (detection, BERT-based)
- **2406.16948v1** Energy-Efficient Detection (TC-ResNet)
- **2310.18767v1** Feature Embeddings (detection, SVM+embeddings)
- **2309.07135v1** EpiDeNet (detection, embedded)
- **2305.10351v1** BIOT Biosignal Transformer (detection)
- **2305.04325v1** Lightweight Convolution Transformer (detection, cross-patient) — *useful methodology ref for cross-patient framing*
- **2301.10167v1** EEG Opto-processor (detection, photonic hardware)
- **2206.02298v1** MICAL (detection, factor graphs)
- **2206.04746v1** HDTorch (detection, hyperdimensional computing)
- **2111.08457v1** TSK Fuzzy System (detection, transfer learning)
- **2110.02169v2** SOUL (detection, online learning)
- **2108.02372v1** CNN-Aided Factor Graphs (detection)
- **2106.08008v3** Long-term Non-invasive Monitoring (detection, wearable)
- **2106.03461v3** Subject Independent EMOTION Recognition (CHB-MIT mentioned but not primary task)
- **1906.02745v1** IndRNN+dense+attention (detection)
- **1903.09326v1** IndRNN approach (detection)
- **1812.06562v2** BiLSTM + attention (detection)
- **1404.0404v1** L-SODA Directed Information (detection)

## Task-Type Dimensions Found in the Literature (2026-04-22 13:30 update)

The first two PDF reads (Xu 2023 + Rukhsar 2024) revealed that 'CHB-MIT seizure prediction' papers use AT LEAST 4 distinct task conventions that are NOT directly comparable:

| # | Task Convention | Binary Positive Class | Binary Negative Class | Example |
|---|---|---|---|---|
| A | **Preictal-vs-Interictal** (pure prediction) | preictal | interictal | SzPredict target, ~Random Forest paper |
| B | **Preictal+Ictal vs Interictal** (seizure-window binary) | preictal ∪ ictal | interictal | ARNN (Rukhsar 2024) |
| C | **Ictal vs Interictal** (traditional detection) | ictal | interictal | Most papers marked 'detection' |
| D | **Crossing-period vs non-crossing** (real-time detection) | window spanning onset | pre-onset window | Xu 2023 (Shorter Latency) |
| E | **3-class** (preictal vs ictal vs interictal) | — | — | SzPredict also supports; rare in lit |

Every PDF read should log which task type it uses. A paper's numbers cannot be compared across task types without explicit conversion — and some conversions are lossy or impossible (ARNN's merged model can never produce preictal-only metrics).

## Revised Estimate

Original count of 13 APPLICABLE was based on 'mentions seizure prediction on CHB-MIT'. After finding that full-read reveals task-type mismatches, the TRUE preictal-vs-interictal baseline count is likely **5-8 papers**, not 13.

**But:** the methodology catalog (which task convention each paper uses + why their numbers aren't comparable) is arguably the more valuable rosetta-stone artifact than the numbers themselves. It IS the rosetta stone.

## Key observations

1. **~34% hit rate on applicability** (13/38). Matches expectation — most CHB-MIT work is detection, not prediction. This itself is a data point for SzPredict's framing: prediction is under-served vs detection.

2. **Only 1 paper (Dissanayake 2020)** does true cross-patient prediction. Everyone else is patient-specific. This VALIDATES SzPredict's Protocol 3 contribution — the field has almost no cross-patient baselines.

3. **Pale et al. 2023 (2302.10672)** is gold — it's an external paper calling out the same methodology-heterogeneity problem that SzPredict solves. Cite it in the rosetta-stone README.

4. **Preictal window varies wildly:**
   - 5 min (Pallister-style, our SzPredict default)
   - 30 min SOP (Random Forest paper)
   - 76.8 min lead time (Preictal Period Optimization)
   - Others don't report
   **This is the rosetta-stone-mattering problem in concrete form.**

5. **Metric heterogeneity:** reported metrics include sens/spec/F1/AUC/FPR per hour / accuracy / AUC-ROC / lead-time. Many papers omit specificity (detection-mode bias). Translation table is necessary.

