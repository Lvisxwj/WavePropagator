# SMILE² release index

Use this file first when locating code, paper files, evidence, or reproducibility notes in the public release package.

## Top-level files

- `README.md` — short release overview.
- `PAPER.md` — compact paper manifest.
- `WORKFLOWS.md` — reproducible workflows and common traps.
- `INDEX.md` — this directory map.

## Logic layer

`logic/` contains the paper-facing substrate. It fixes terminology, notation, and evidence boundaries.

- `logic/problem.md` — CASSI problem framing and end-to-end scope.
- `logic/concepts.md` — method names, term definitions, and safe wording.
- `logic/algorithm.md` — CASSI forward model, CASSI Prep, SFEstimator/SFEvolver signal flow, SWP equations, and training loss.
- `logic/claims.md` — key claims and supporting evidence.

Use `logic/algorithm.md` as the notation source of truth for figures and LaTeX.

## Code layer

`src/repo/` is the current implementation.

- `train.py` — main YAML-driven training entry.
- `smoke_test.py` — lightweight forward/backward sanity check.
- `test_real.py` — TSA-real reconstruction entry.
- `batch_scan.py` — throughput/OOM batch-size probe.
- `dataset.py`, `loss.py` — data loading, RMSE loss, PSNR/SSIM/SAM metrics.
- `configs/` — release configs.
  - `smile_s.yaml`, `smile_m.yaml`, `smile_l.yaml` — main model family.
  - `ablations/` — paper ablation configs.
  - `experimental/` — preserved exploratory configs.
- `model/` — model implementation.
  - `smile.py` — SMILE² wrapper and CASSI-side preprocessing.
  - `spectral_wave_propagator.py` — Spectral Wave Propagator and SWP block.
  - `spectral_field_components.py` — SFEstimator/SFEvolver building blocks.
  - `spatial_content_modulation.py` — spatial-content modulation block.
  - `cassi_operators.py` — mask-conditioned CASSI utilities.
- `scripts/` — launch/evaluation/status helper scripts that belong to the code package.
- `analysis/` — mechanism diagnostics and analysis scripts.

`src/tools/` contains only public helper notes in this release. Private remote helpers and local one-off figure scripts are intentionally excluded.

## Evidence layer

- `evidence/checkpoints/SMILE-M/` — released SMILE²-M checkpoint backup and provenance note.
- `evidence/results/` — SOTA/ablation CSVs, markdown tables, citation planning, and narrative evidence.
- `evidence/figures/` — generated visual assets and mechanism evidence used during writing.

Do not fill numbers from memory; use evidence files or regenerated evaluation outputs.

## Paper layer

- `paper/aaai2027/smile2_aaai2026.tex` — active English AAAI manuscript.
- `paper/aaai2027/smile2_aaai2026.bib` — active bibliography.
- `paper/aaai2027/smile2_aaai2026.pdf` — compiled draft.
- `paper/aaai2027/figures/final/` — final figure PDFs referenced by TeX.
- `paper/aaai2027/author_kit/` — official AAAI author kit.
- `paper/feedback/` — professor/peer feedback snapshots used during revision.


