# SMILE² / SpectralWavePropagator

Clean release package for **SMILE²: Spectral Modulated Imaging via Learned Estimation-Evolution**.

SMILE² is an end-to-end CASSI hyperspectral reconstruction framework built around a Spectral Wave Propagator (SWP), which models spatial-spectral feature propagation through a Fourier-domain closed-form damped wave operator.

## Layout

```text
release/
├── README.md                 # this overview
├── PAPER.md                  # short paper manifest
├── INDEX.md                  # detailed file map
├── WORKFLOWS.md              # reproducibility and maintenance notes
├── logic/                    # paper-facing notation, claims, concepts, algorithm
├── paper/aaai2027/           # AAAI manuscript source, bibliography, figures, PDF
├── evidence/                 # selected metrics, figures, and checkpoint backup
└── src/
    ├── environment.md        # dependency and runtime notes
    ├── repo/                 # SMILE² training/evaluation implementation
    └── tools/                # public helper notes; private/local one-off tools excluded
```

## Code entry points

Current implementation is under `src/repo/`.

- `train.py` — YAML-driven training.
- `smoke_test.py` — quick forward/backward check.
- `test_real.py` — TSA-real inference entry.
- `batch_scan.py` — batch-size throughput/OOM probe.
- `model/smile.py` — SMILE² wrapper and CASSI preprocessing.
- `model/spectral_wave_propagator.py` — SWP and SWP block implementation.
- `model/spectral_field_components.py` — SFEstimator/SFEvolver components.
- `model/cassi_operators.py` — mask and CASSI operator utilities.

## Main configs

- `src/repo/configs/smile_s.yaml`
- `src/repo/configs/smile_m.yaml`
- `src/repo/configs/smile_l.yaml`

Ablations are under `src/repo/configs/ablations/`; exploratory historical configs are under `src/repo/configs/experimental/`.

## Checkpoint

A local backup of the released SMILE²-M checkpoint is stored at:

```text
evidence/checkpoints/SMILE-M/SMILE-M_best_psnr.pth
```

Private server paths and credentials are intentionally excluded.

## Paper

The active English manuscript is:

```text
paper/aaai2027/smile2_aaai2026.tex
paper/aaai2027/smile2_aaai2026.pdf
```


