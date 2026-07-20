# SMILE² reproducible workflows and known traps

This file exists to prevent repeated wheel-reinvention. Before running LaTeX, remote/A800 checks, table generation, or figure exports, skim the relevant section.

## 1. General Windows rules

- Prefer PowerShell with `-LiteralPath` for paths containing spaces or Chinese characters.
- Prefer `python -X utf8` for scripts that read/write Markdown, TeX, CSV, or BibTeX.
- When writing text files from PowerShell, use explicit UTF-8 encoding.
- Do not paste long remote commands with pipes directly into PowerShell. Write a `.sh` script under `src/tools/remote/` or a temp script, upload it, then execute it remotely.
- Do not delete helper scripts after a run; keep them for debugging and reuse.

## 2. LaTeX / AAAI workflow

Active manuscript directory:

```text
paper/aaai2027/
```

Important files:

- `smile2_aaai2026.tex` — main paper.
- `smile2_aaai2026.bib` — active BibTeX.
- `figures/final/` — figure PDFs referenced by TeX.
- `author_kit/` — official AAAI kit. Use it to check formatting rules.

Recommended compile/check workflow:

1. Work inside `paper/aaai2027/`.
2. Keep figures as PDF when possible; use tight PDFs for SVG-derived figures.
3. If an SVG/PDF figure becomes huge or crops badly, regenerate through `src/tools/figures/export_svg_tight_pdf.py` or inspect the PDF bounding box before changing TeX layout.
4. If references show `[??]`, check in this order:
   - the citation key exists in `smile2_aaai2026.bib`;
   - the key spelling in `.tex` matches exactly;
   - BibTeX has been rerun after LaTeX;
   - no stale `.aux/.bbl` is masking the error.
5. For Windows encoding issues, rerun helper scripts with `python -X utf8` and avoid non-UTF8 shell redirection.

Paper wording rule: the `.tex` should contain reviewer-facing language only. Internal phrases like “current checkpoint”, “to be filled later”, “we know this is unfinished”, “see local md”, or “temporary” belong in trace/evidence notes, not in the manuscript.

## 3. A800 / remote workflow

Primary rule: do not improvise remote commands. Use existing helpers under:

```text
src/tools/remote/
```

Most useful scripts:

- `a800_status.ps1` / `a800_status.py` — fixed status collection entry. Use first for logs/GPU state.
- `a800_exec.py` — execute a remote command through the project helper stack.
- `a800_upload_and_run.py` — upload a script and run it remotely.
- `a800_transfer.py` / `a800_fetch.py` — transfer results back.
- `a800_parse_current_logs.sh` / `a800_parse_ablation_best.py` — parse training logs.
- `a800_eval_smile_s_l.sh`, `a800_export_smile_s_l_eval.sh` — SMILE-S/L evaluation/export helpers.
- `a800_measure_runtime.sh`, `a800_runtime_probe.sh` — latency/memory/runtime probes.

Known traps:

- Some remote project paths require `sudo`; ordinary `screen -ls` may not show root screens.
- GPU process state, screen state, log timestamp, and result CSV should be checked together. Screen alone is not enough.
- PowerShell can break remote grep/pipe/quote syntax. Use uploaded `.sh` scripts for anything nontrivial.
- PyTorch 2.6 may require `torch.load(..., weights_only=False)` for training checkpoints containing RNG state.
- TSA simulation data can contain duplicate `.mat/.npy` views; paper tables should use the folded 10-scene results, not accidental 20-entry duplicates.
- Store secrets outside committed files. Do not hard-code passwords in scripts or README files.

Status outputs should be saved under evidence or trace, not only read from terminal.

## 4. SOTA table workflow

Primary generated table source:

```text
evidence/results/sota_table.md
```

Primary CSV source:

```text
evidence/results/sota_csv/
```

Helper:

```text
src/tools/paper/build_sota_table.py
```

Rules:

- Do not fill numbers from memory.
- For SMILE-S/M/L, use the current per-scene CSV and summary CSV in `evidence/results/sota_csv/`.
- If a method has no public checkpoint/output, table values may come from the paper, but this must be stated in the caption or experiment protocol.
- Keep method categories consistent with the paper table style: CNN / Transformer / Wave equation / Optimization, etc.
- Keep SAM blank or marked unavailable when a paper does not report it and no output/checkpoint exists.

## 5. Figure workflow

Final paper figures live in:

```text
paper/aaai2027/figures/final/
```

Figure source and generated drafts may live in:

```text
paper/figures/
evidence/figures/
src/tools/figures/
```

Rules:

- Final LaTeX should reference final PDFs, not random temp PNGs.
- For mechanism figures, check notation against `logic/algorithm.md` before exporting.
- For SOTA visual comparisons, keep the model set consistent across scene, curve, and real-result figures.
- Use `SMILE²` naming consistently in final labels.
- If a figure is visually useful but not final, keep it under evidence/trace rather than `paper/aaai2027/figures/final/`.

## 6. Checkpoint workflow

Local checkpoint backup:

```text
evidence/checkpoints/
```

Rules:

- SMILE-S/M/L checkpoint backups are evidence, not source code.
- Large checkpoints should not be copied into sanitized `published/` unless explicitly intended.
- When comparing variants, record config, checkpoint path, epoch, PSNR, SSIM, SAM, Params, and FLOPs together.
- Prefer exported CSVs for table updates.

## 7. Deep unfolding boundary

The current paper is an end-to-end SMILE² paper. Deep unfolding is preserved at `../deep_unfolding/` because it is scientifically useful, but DU language should not leak into the current method unless deliberately discussed as related background or internal provenance.

Use `../deep_unfolding/` for future extensions, teacher/history checks, and old failure analysis.

