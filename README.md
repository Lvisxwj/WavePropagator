# SMILE2

Clean project package for the SMILE2 CASSI reconstruction paper and code.

This is the active Git-tracked project root. Legacy DU code, external SOTA
repositories, old experiments, and bulky figure drafts are kept outside this
directory under `../_legacy/` unless they are explicitly needed by the current
paper/code workflow.

## Layout

```text
SMILE2/
├── AGENTS.md              # project operating rules and A800 runbook memory
├── PAPER.md               # short project manifest
├── logic/                 # paper-facing concepts, claims, problem, algorithm
├── paper/
│   ├── aaai2027/          # AAAI LaTeX source, bibliography, compiled draft
│   ├── feedback/          # professor feedback and Chinese working draft refs
│   └── figures/           # source/final figure assets used by the paper
├── evidence/
│   ├── checkpoints/       # local backups of SMILE-S/M/L checkpoints
│   ├── figures/           # generated mechanism/visual comparison assets
│   └── results/           # SOTA/ablation CSVs and markdown tables
└── src/
    ├── repo/              # current E2E implementation restored from A800
    ├── tools/             # reusable local/A800/eval/figure helper scripts
    └── environment.md     # reproducibility notes
```

## Current code entry

Current training/evaluation code is in `src/repo/`.

Important entry points:

- `train.py`
- `test_real.py`
- `train_distill.py`
- `batch_scan.py`
- `smoke_test.py`
- `configs/runtime_friend/`
- `model/`

Root-level launch/status/eval helper scripts are intentionally moved under
`src/repo/scripts/` so that the code root stays readable.

## A800 helpers

A800 connection/status/export scripts are centralized in `src/tools/remote/`.

The pinned Paramiko dependency used for A800 helper scripts is stored in
`.codex_deps/paramiko_a800/`. Use this folder via `PYTHONPATH` when the system
Paramiko is too old.

## Additional local research notes

- 	race/: development history, side analyses, old drafts, and experiment-audit notes. These are useful context but not the canonical method description.

## Start here

- INDEX.md — detailed directory/file map for the active repo.
- WORKFLOWS.md — fixed workflows for LaTeX, A800, SOTA tables, figures, and checkpoints.
- AGENTS.md — long internal run-state and historical-project reference.



