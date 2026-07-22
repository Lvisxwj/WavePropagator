# SMILE² reproducible workflows and known traps

This file records the stable public workflows for the release package. It intentionally excludes private server credentials, private absolute paths, and one-off local figure hacks.

## 1. General Windows rules

- Use PowerShell `-LiteralPath` for paths containing spaces or Chinese characters.
- Use `python -X utf8` for scripts that read/write Markdown, TeX, CSV, or BibTeX.
- Avoid non-UTF8 shell redirection when touching manuscript files.
- Keep reusable scripts; do not delete a script simply because one run has finished.

## 2. LaTeX / AAAI workflow

Active manuscript directory:

```text
paper/aaai2027/
```

Important files:

- `smile2_aaai2026.tex` — main paper.
- `smile2_aaai2026.bib` — active BibTeX.
- `figures/final/` — final figure PDFs referenced by TeX.
- `author_kit/` — official AAAI kit.

Recommended workflow:

1. Work inside `paper/aaai2027/`.
2. Keep final figures as tight PDFs where possible.
3. If references show `[??]`, check citation keys, rerun BibTeX/LaTeX, and remove stale `.aux/.bbl` only when necessary.
4. Keep reviewer-facing language in the `.tex`; internal notes belong in evidence or logic files, not the manuscript.

## 3. Training and evaluation workflow

Code directory:

```text
src/repo/
```

Typical commands:

```bash
python smoke_test.py
SMILE_CONFIG=configs/smile_m.yaml python train.py
SMILE_CONFIG=configs/smile_m.yaml python batch_scan.py
```

Main configs:

```text
configs/smile_s.yaml
configs/smile_m.yaml
configs/smile_l.yaml
```

Ablation configs are under `configs/ablations/`. Exploratory historical configs are under `configs/experimental/`.

## 4. SOTA table workflow

Primary evidence source:

```text
evidence/results/
```

Rules:

- Do not fill numbers from memory.
- If a method has no public checkpoint/output, table values may come from the paper, but the experiment protocol or caption should say so.
- Keep SAM unavailable when a paper does not report it and no output/checkpoint exists.
- Keep method categories consistent with the manuscript table style.

## 5. Figure workflow

Final manuscript figures live in:

```text
paper/aaai2027/figures/final/
```

Rules:

- Final LaTeX should reference final PDFs, not temporary PNGs.
- Check notation against `logic/algorithm.md` before exporting mechanism figures.
- Keep the model set consistent across scene comparison, spectral curve, and real-result figures.
- Use `SMILE²` naming consistently in final labels.

## 6. Checkpoint workflow

Released checkpoint backup:

```text
evidence/checkpoints/SMILE-M/SMILE-M_best_psnr.pth
```

Rules:

- Checkpoints are evidence, not source code.
- Record config, checkpoint path, epoch, PSNR, SSIM, SAM, Params, and FLOPs together when comparing variants.
- Keep private server paths out of committed provenance files.

## 7. Deep unfolding boundary

This release is for the end-to-end SMILE² route. Deep unfolding experiments remain useful scientific background, but DU training code and private histories are not part of this sanitized release package.
