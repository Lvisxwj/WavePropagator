# SMILE² main repository index

This is the working index for the current SMILE² project. Use this file first when you need to find code, paper files, figures, results, or reusable workflows.

## First entry points

- `README.md` — short project overview and semantic layout.
- `INDEX.md` — this file; detailed map of directories and key files.
- `WORKFLOWS.md` — reproducible workflow notes for LaTeX, A800, figures, SOTA tables, and checkpoints.
- `AGENTS.md` — long internal project memory and run-state reference. Use it before touching remote experiments or old branches.
- `PAPER.md` — compact paper manifest.

## Cognitive / paper-logic layer

`logic/` contains the stable writing substrate. It should define concepts and formulas, not temporary TODOs.

- `logic/problem.md` — problem framing, why CASSI is hard, and the E2E/DU boundary.
- `logic/concepts.md` — naming, term definitions, and boundaries.
- `logic/algorithm.md` — fixed notation, CASSI forward model, CASSI Prep, SFEstimator/SFEvolver signal flow, SWP equations, and training loss.
- `logic/claims.md` — falsifiable paper claims and what evidence supports each claim.

Use `logic/algorithm.md` as the source of truth for symbols in figures and LaTeX.

## Physical / code layer

`src/` stores implementation and reusable utilities.

- `src/environment.md` — environment/reproducibility notes.
- `src/repo/` — current E2E SMILE² training/evaluation code.
  - `train.py` — main training entry.
  - `train_distill.py` — historical KD/distillation entry; not current paper mainline.
  - `test_real.py` — real-data inference/evaluation entry.
  - `dataset.py`, `loss.py` — dataset loading and metrics/loss.
  - `model/` — model implementation.
    - `e2e.py` — SMILE² wrapper and CASSI-side preprocessing logic.
    - `wpo3d.py` — SWP / wave-kernel implementation.
    - `mask_ops.py` — mask operations and gate helpers.
    - `cmb.py` — old/modular mixing blocks, only use when explicitly needed.
  - `configs/` — YAML experiment configs.
  - `analysis/` — mechanism diagnostics and analysis scripts.
  - `scripts/` — training/launch helpers bundled with repo code.
  - `real_results/` — real-data generated outputs.
- `src/tools/` — helper scripts outside the model package.
  - `remote/` — A800/remote execution, status, transfer, runtime probing.
  - `eval/` — per-scene and checkpoint evaluation helpers.
  - `figures/` — figure generation/export helpers.
  - `paper/` — SOTA table and LaTeX-adjacent helpers.

## Evidence layer

`evidence/` stores data that supports the paper and figures.

- `evidence/checkpoints/` — local backup of SMILE-S/M/L checkpoints and checkpoint README.
- `evidence/results/` — tables, CSVs, ablation drafts, citation planning, and narrative notes.
- `evidence/figures/` — generated figure assets and older temp visual outputs.

Use evidence files to update paper tables; do not edit paper numbers from memory.

## Paper layer

`paper/` stores manuscript assets.

- `paper/aaai2027/` — active English AAAI manuscript.
  - `smile2_aaai2026.tex` — main TeX file.
  - `smile2_aaai2026.bib` — active BibTeX file.
  - `figures/final/` — final figure PDFs used by the paper.
  - `author_kit/` — official AAAI author-kit copy.
- `paper/figures/` — figure source/export materials outside the final LaTeX folder.
- `paper/feedback/` — professor/peer feedback snapshots.

## Trace / exploration graph

`trace/` keeps valuable development history and context without polluting the current paper narrative.

- `trace/e2e_development/` — E2E plans, diagnostics, paper-planning notes.
- `trace/mamba_side_analysis/` — Mamba/AFFB analysis; related-work support only, not current method.
- `trace/old_paper_drafts/` — previous paper drafts and tables.
- `trace/early_waveprop_notes/` — early WPO/SWAP notes and handoff files.
- `trace/environment_notes/` — old root environment/agent notes.
- `trace/pre_reorg_archive/` — pre-reorganization snapshot fragments.

## What is outside this repo directory

- `../deep_unfolding/` — important DU branch, separated from the E2E mainline.
- `../reference/` — external papers, SOTA code, author kits.
- `../published/` — sanitized collaborator/GitHub release copy.
- `../_legacy/` — low-priority old leftovers only.

