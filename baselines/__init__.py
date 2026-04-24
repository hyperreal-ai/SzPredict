"""SzPredict reference baselines.

Three tiers, deliberately simple, used to establish floor/ceiling context:
  - baseline_random: uniform random 3-class. Pure floor.
  - baseline_majority: always predict interictal. Shows the degenerate trap.
  - baseline_cnn: minimal 1D-CNN on raw windows. Legitimate non-degenerate reference.

All three use ONLY standard components. No wavelets, no FFT tokenizer, no
proprietary architecture. The benchmark is open; the models that compete on it
are yours to keep.
"""
