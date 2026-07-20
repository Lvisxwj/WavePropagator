# SMILE2 E2E Code

Current single-pass CASSI reconstruction implementation restored from the A800
`e2e.v1` project.

## Main entry points

- `train.py`: YAML-driven single-GPU training.
- `test_real.py`: TSA-real evaluation/visualization.
- `train_distill.py`: historical distillation/curriculum entry; kept for
  reproducibility, not the main paper route.
- `batch_scan.py`: batch-size throughput/OOM scan.
- `smoke_test.py`: quick forward/backward check.

## Core code

- `model/`: SMILE2 model, SWP/SFEstimator/SFEvolver code, mask operations.
- `configs/`: experiment configs.
- `configs/runtime_friend/`: current A800/friend-server runtime configs,
  including SMILE-S/M/L and ablation variants.
- `dataset.py`, `loss.py`: data loading, RMSE loss, PSNR/SSIM/SAM metrics.

## Supporting scripts

To keep the root readable, launch/eval/status helper scripts live under:

- `scripts/launch/`
- `scripts/eval/`
- `scripts/status/`

Friend-server dataset/config helper scripts remain under `scripts/`.

## Notes

- Checkpoints, logs, datasets, and generated arrays are intentionally not
  committed here.
- Local SMILE-S/M/L checkpoint backups are stored outside the code tree at
  `../../evidence/checkpoints/`.

