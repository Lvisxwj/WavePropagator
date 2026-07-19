# E207 Capacity spectral mechanism analysis

This directory contains post-hoc diagnostics only. It does not modify or train
the model.

## Analyses

1. `frequency_response`: analytical initial-displacement transfer of every WPO
   layer at zero spatial frequency, plus its equivalent spectral kernel.
2. `empirical_mixing_map`: finite-difference response of the complete nonlinear
   backbone to a small, localized perturbation in each input spectral band.
3. `occlusion_recovery`: recovery after removing known band contributions from
   the simulated CASSI measurement; it also separates adjacent and distant
   spectral context.

Every figure is stored as PNG and PDF. Every plotted result has a CSV or NPY
counterpart. Interpretation limitations are recorded in `manifest.json`.

Run immediately:

```bash
cd src/repo
SMILE_CONFIG=analysis/config.yaml python analysis/analyze_spectral.py --config analysis/config.yaml
```

Queue after K1 on GPU1:

```bash
screen -dmS spectral_analysis -L -Logfile analysis/logs/spectral_analysis.log \
  bash analysis/run_after_k1.sh
```



