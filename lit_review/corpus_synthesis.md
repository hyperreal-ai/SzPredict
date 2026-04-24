# SzPredict Lit Review — Corpus Synthesis

**Date:** 2026-04-22
**Source corpus:** 19 papers from `~/ai/_datasets/arxiv/arxiv_papers.db`, CHB-MIT-mentioning
**Analysis method:** Source PDF → Claude structured extraction → `.txt` in `pdfs/` → Claude synthesis
**Status:** Individual paper `.txt` extractions saved. This file captures the META-findings across the 19.

---

## 1. Task-type conventions (4 distinct, not comparable)

| # | Convention | Positive class | Negative class | Example papers |
|---|---|---|---|---|
| A | **Preictal-vs-Interictal** (pure prediction) | preictal (within window before onset) | interictal | **SzPredict target**, Wang 2021, Rasheed 2020, Ben Messaoud 2020, Godoy 2022, Koutsouvelis 2024, Dissanayake 2020, Shu 2024, Bhattacharya 2022, Chen 2024 (pred), Lu 2024, Wang 2024, Li 2022, Xu 2021, Zhao 2022 |
| B | **Preictal+Ictal vs Interictal** ('seizure window' binary) | preictal ∪ ictal | interictal | Rukhsar 2024 (ARNN) |
| C | **Ictal vs Interictal** (traditional detection) | ictal | interictal | Chen 2024 (detection), many out-of-corpus detection papers |
| D | **Crossing-period vs non-crossing** (real-time detection) | window straddling onset | pre-onset window | Xu 2023 |
| E | **3-class** (preictal / ictal / interictal) | — | — | Liu 2021 supports; SzPredict supports; rare in literature |

**Key consequence:** Numbers across conventions are NOT directly comparable. ARNN's 99.96% accuracy merges preictal+ictal; SzPredict's Protocol 3 distinguishes them. Translation table needs task-type as a primary axis.

---

## 2. The 10+ Rosetta-Stone Dimensions

Every comparable paper entry needs these axes populated. Missing values = flagged as UNKNOWN in translation.

1. **Task type** (A / B / C / D / E above)
2. **Dataset version** (raw CHB-MIT / pre-processed derivative [46] Pale2023 / Meta-EEG [47] / SRH-LEI / Kaggle / Freiburg)
3. **SPH — Seizure Prediction Horizon** (5, 10, 30, 60 min, implicit-0, N/A)
4. **SOP — Seizure Occurrence Period** (30, 50, 60 min typical)
5. **Preictal window duration** (1, 3, 10, 15, 30, 60 min — six distinct values in corpus)
6. **Split protocol** (patient-specific / LOPO / cross-patient-fixed / 10-fold-random / LOSOCV / temporal-LOOCV)
7. **Feature space** (raw EEG / STFT / Wavelet / MFCC / hand-engineered + PCA / hand-engineered + other)
8. **Augmentation strategy** (none / window overlap / SMOTE / GAN / DDPM / segment shuffle / patient-indep representation)
9. **Channel count** (23 / 22 / 18-common / 16-Kaggle / 8-selected / 6-hypothetical)
10. **Patient cohort size** (5, 6, 7, 13, 15, 19, 20, 22, 24)
11. **Metrics reported** (Sens / Spec / Acc / AUC / F1 / FPR-per-hr / prediction time / balanced acc)
12. **Re-implementation discipline** (Y = re-ran baselines under matched protocol / N = cited others' self-reported numbers)
13. **Post-processing** (none / sustainability / k-of-n / Hanning / WMV / refractory / accumulative-probability)
14. **Calibration type** (cohort / per-patient thresholds / per-patient fine-tune / none)
15. **Deployment form-factor** (cloud-ok / wearable / on-implant / unspecified research)

**Highest-impact for non-comparability:** Task type (axis 1) + Preictal window (5) + Split protocol (6). These three alone account for most "apples-to-oranges" in the literature.

---

## 3. Convergence: Temporal-smoothing post-processing (6 independent groups)

The most striking methodological consensus in the corpus. Six separate groups independently converged on 'require sustained positive prediction before alarm' as a mandatory FAR reducer:

| Paper | Method |
|---|---|
| Dissanayake 2020 | Hanning window smoothing |
| Ben Messaoud 2020 | Sustainability rule: T* threshold for n* consecutive epochs |
| Liu & Richardson 2021 | Weighted Majority Voting (WMV) with consecutive-prediction bonus — most formally parameterized |
| Bhattacharya 2022 | k-of-n rule (k=8, n=10 over 300 s) |
| Xu 2023 | Accumulative probability (Eq. 5) |
| Shu 2024 | k-of-n + 30-min refractory period |

**Quantitative evidence (Liu 2021 Table):**
- Detection FAR: 0.745/h → 0.169/h = **4.4× reduction** via WMV
- Prediction FAR: 2.341/h → 0.710/h = **3.3× reduction**

**Preprint implication:** this is a FIELD-LEVEL consensus hiding in plain sight. Prominent treatment in the preprint's methods section. Frame as 'the field agrees on this, but every paper implements it differently — SzPredict standardizes the comparison.'

---

## 4. Convergence: Data-scarcity mitigation (5 distinct strategies)

CHB-MIT's small-sample problem has driven the field to explore 5 distinct routes:

| Strategy | Representative papers |
|---|---|
| **Architectural efficiency** | Rukhsar 2024 (ARNN windowed attention), Wang 2021 (dilated 3D), Zhao 2022 (binary) |
| **Patient-independent representation learning** | Dissanayake 2020 (Siamese + contrastive) |
| **Synthetic data generation** | Rasheed 2020 (DCGAN), Shu 2024 (DDPM) |
| **Overlap-based oversampling** | Xu 2021, most CenBRAIN papers, Wang 2021, Liu 2021 |
| **Feature engineering + classical ML** | Ben Messaoud 2020 (RF + 735 features), Li 2022 (PCA of 176 features + memristive CNN) |

**Preprint implication:** organize a review section around 'what problem is each paper actually solving?' instead of by architecture family. This is a better approach than 'a survey of seizure prediction architectures.'

---

## 5. Compact-deployment cluster (4 approaches, 2024-era)

Four routes to <50K parameters, wearable/implant-target seizure prediction:

| Approach | Paper | Params | Headline |
|---|---|---|---|
| Binarized digital CNN | Zhao 2022 BSDCNN | 672K binary | 25.5× compute reduction |
| Full-precision analog RRAM inference | Li 2022 | 10.8K | 8.12 µJ/prediction (simulated) |
| RRAM dynamics as feature extractor | Wang 2024 | minimal 2-layer | **1.515 µJ/prediction (simulated)** |
| Spiking Transformer neuromorphic | Chen 2024 | 9.9K/40.3K | >10× op reduction |
| Channel selection + Mamba | Lu 2024 | 21.2K | 8 channels (22→8) |

**Missing from the field:** the (rigorous evaluation + hardware-efficient + high accuracy) combination. Every hardware-focused paper in corpus skips rigorous evaluation. Every rigorous-evaluation paper (Ben Messaoud, Koutsouvelis) doesn't optimize for deployment. **SzPredict's positioning is to force the combination.**

---

## 6. Measured-vs-simulated silicon gap (Liu 2021 vs Wang 2024)

| Paper | Platform | Energy/inference | Measured? |
|---|---|---|---|
| Liu & Richardson 2021 | ARM Cortex-M4 COTS | 102-1,377 µJ | **Measured** |
| Wang 2024 | RRAM + 22nm CMOS | 1.515 µJ | Simulated |
| Li 2022 parallelized | RRAM + 22nm CMOS | 8.12 µJ | Simulated |
| Zhao 2022 | Binary digital ASIC | Not quantified | Simulated |
| Chen 2024 | Neuromorphic | Ops only, not µJ | Calculated |

**Gap:** ~100× between COTS (measured) and ASIC/RRAM (simulated). Quantifies the 'custom silicon is needed for implant deployment' argument with actual numbers.

**Only Liu 2021 has silicon measurements in the entire corpus.** All other hardware claims are simulations.

---

## 7. Patient cohort size → accuracy inverse correlation

Smaller cohort usually correlates with higher headline accuracy. Approximate pattern:

| Cohort | Papers | Typical Sens |
|---|---|---|
| 5-7 patients | Li 2022, Xu 2021, Wang 2021, Zhao 2022 | 94-99% |
| 13 patients | Rasheed 2020, Shu 2024, Wang 2024 | 91-96% |
| 15 patients | Liu 2021 | 90-97% |
| 19 patients | Xu 2023, Koutsouvelis 2024 | 90-99% |
| 20 patients | Ben Messaoud 2020 | 82-89% |
| 22-24 (full) | Dissanayake 2020, Chen 2024, Lu 2024 | 88-96% |

**Correlation effect:** picking only 'predictable' patients inflates sensitivity. Papers using aggressive exclusion rules (<10 seizures/day, ≥3 seizures, etc.) systematically outperform full-cohort evaluations.

**SzPredict response:** fixed cross-patient split with ALL qualifying patients. Pale et al.'s rubric demands this.

---

## 8. Preictal duration spectrum (complete across corpus)

| Duration | Papers |
|---|---|
| 1 min | Shu 2024 |
| 3 min | Bhattacharya 2022, Liu 2021 |
| 10 min | Rasheed 2020, Ben Messaoud 2020 (τ=10min) |
| 15 min | Chen 2024 |
| 30 min | Xu 2021, Zhao 2022, Wang 2021, Li 2022, Wang 2024, Lu 2024 |
| 60 min | Dissanayake 2020, Godoy 2022, Koutsouvelis 2024 |

**Critical finding:** 60-min preictal is an EASIER task than 5-min preictal. Papers quoting 90%+ sensitivity at 60-min are not comparable to SzPredict's 5-min target. Translation must be by duration.

**Godoy's 2022 clean finding:** 60-min preictal beats 30-min beats 15-min on all architectures (patient-specific). But Dissanayake 2020's cross-patient finding is OPPOSITE: accuracy drops as horizon extends. SzPredict position: patient-specific training can exploit longer preictal; patient-independent cannot.

---

## 9. Paper role map (the preprint's 8-section structure)

In Pale review, line 100-107; Koutsouvelis review, line 297-305; Each paper in corpus maps to exactly one role:

| Section | Role | Paper |
|---|---|---|
| 1 | **Methodology critique foundation** | **Pale et al. 2023** (paper #5) |
| 2 | **Dataset provenance problem** | Handa 2023 (survey) |
| 3 | **Task-taxonomy problem** | Dissanayake 2020 |
| 4 | **Classical-ML methodological exemplar** | Ben Messaoud 2020 |
| 5 | **Modern-DL methodological exemplar** | **Koutsouvelis 2024** |
| 6 | **Probabilistic-output reformulation** | Xu 2023 |
| 7 | **Latency-measurement problem** | Xu 2023 |
| 8 | **Architecture/feature trade-off** | Rukhsar 2024 |
| 9 | **Data-scarcity solutions (generative)** | Rasheed 2020, Shu 2024 |
| 10 | **Representation-learning solutions** | Dissanayake 2020, Rukhsar 2024 |
| 11 | **Hardware-deployment spectrum** | Li 2022, Zhao 2022, Wang 2024, Chen 2024, Liu 2021 |
| 12 | **Headline-chasing foil (cautionary)** | Xu 2021, Bhattacharya 2022, various CenBRAIN |

**Preprint thesis:** the field has segmented into these roles, but no single benchmark can evaluate them on equal footing. SzPredict's Protocols 1-4 + standardized metrics + clinical-utility framing (Protocol 4) force the comparison. The methodology-heterogeneity problem is the real story; the architectural differences are downstream of it.

---

## 10. Key citations for the preprint's critical core

- **Pale et al. 2023** — external validation of SzPredict's methodology critique
- **Koutsouvelis 2024** — ally paper (CIOPR metric, 76.8-min prediction time, 19 patients LOSOCV)
- **Ben Messaoud 2020** — classical-ML rigor example; shows strict protocol produces lower but honest numbers
- **Dissanayake 2020** — 'patient-independent' vs 'LOPO' conflation critique; 58.55% LOPO vs 91.54% 10-fold headline
- **Liu & Richardson 2021** — only measured-silicon baseline in corpus; quantifies COTS-vs-ASIC gap
- **Xu 2023** — 'crossing period' framing; self-aware Line 97 quote: 'Don't compare directly to older papers' sensitivity numbers'

## 11. Five most prominent conceding quotes (for preprint's critique section)

1. **Xu 2023** (paper 2301.03465, line 97 of extraction):
   > 'Sensitivity caveat: their reported sensitivity is strictly crossing-period sensitivity. Post-crossing, sensitivity = 100%. Don't compare directly to older papers' sensitivity numbers without this note.'

2. **Pale et al. 2023** (line 56 of extraction):
   > 'L1O demonstrates that training on future data leads to overestimated performance and should be avoided.'

3. **Pale et al. 2023** (line 50):
   > 'Balanced-subset FAR is an extrapolation artifact.'

4. **Dissanayake 2020** (line 97 of extraction):
   > 'LOPO generalization is genuinely hard. The average LOPO accuracy (58.55%) is close to chance... Most "patient-independent" papers conflate [patient-independent with zero-shot to unseen patients].'

5. **Rukhsar 2024** (line 122 of extraction):
   > 'The class-balance procedure merges preictal + ictal into the positive class on CHB-MIT (not the standard ictal-only definition used by Xu et al. and most of the field). This is a non-trivial labeling difference worth flagging when comparing CHB-MIT numbers across papers.'

## 12. SzPredict's positioning

**What:** The standardized unifying benchmark that measures what actually matters clinically.

**Why needed:** The field has ≥4 task conventions, 6 preictal durations, 10+ non-normalized heterogeneity dimensions, and a methodology crisis (Pale 2023). Published numbers are mostly incomparable.

**How SzPredict differs:**
- Fixed cross-patient protocol (Protocol 3) — full cohort, no cherry-picking
- Protocol 4 (clinical utility / transition timing) — measures lead time and specificity under continuous EEG, not classification accuracy
- Open infrastructure — anyone submits results, metrics computed consistently
- Honest about its own model's performance — baselines degenerate on cross-patient eval, documented
- Methodology rubric — scores each entry on Pale's 6 axes, flags missing methodology

**Who it serves:**
- **Researchers:** a place to compete on equal footing
- **Hardware designers:** a rigorous target to optimize for
- **Clinicians:** metrics that match what a wearable device actually needs to do

