# SMILE² E2E Code

This directory contains the current end-to-end CASSI reconstruction implementation used by SMILE².

## Main entry points

- `train.py` — YAML-driven training.
- `smoke_test.py` — quick forward/backward check.
- `test_real.py` — real-data inference entry.
- `batch_scan.py` — batch-size throughput/OOM scan.

## Core code

- `model/smile.py` — SMILE² wrapper and CASSI Prep.
- `model/spectral_wave_propagator.py` — Spectral Wave Propagator (SWP) and SWP block.
- `model/spectral_field_components.py` — SFEstimator/SFEvolver building blocks.
- `model/spatial_content_modulation.py` — spatial-content modulation block.
- `model/cassi_operators.py` — mask-conditioned CASSI utilities.
- `dataset.py`, `loss.py` — data loading, RMSE loss, PSNR/SSIM/SAM metrics.

## Configs

- `configs/smile_s.yaml`
- `configs/smile_m.yaml`
- `configs/smile_l.yaml`
- `configs/ablations/`
- `configs/experimental/`

## Scripts

Launch/evaluation/status helpers live under:

- `scripts/launch/`
- `scripts/eval/`
- `scripts/status/`

## Notes

- Datasets, logs, generated arrays, and most checkpoints are intentionally not committed.
- The released SMILE²-M checkpoint backup is stored at `../../evidence/checkpoints/SMILE-M/`.


