# SMILE² E2E Name Mapping

> Current authority: 2026-07-16  
> This file only defines the **active E2E paper vocabulary**. Historical explorations are moved to `legacy.md`.

## 1. Core naming

| Paper term | Short name | Meaning | Code-level anchor |
|---|---|---|---|
| **SMILE²** | — | Single-pass CASSI reconstruction via learned estimation and physical evolution | `E2ESMILE` |
| **CASSI Field Preparation** | **CASSI Prep** | Construct the dirty initial field and residual-adjoint cue from CASSI measurement and mask | shift-back, forward/adjoint operators |
| **Spectral Field Estimator** | **SFE** | Estimate a cleaner spectral field from the dirty CASSI initialization; logically includes the SFE stem and the first SWP-based U-Net | `MeasurementAwareInitialStateStem` + first `WaveMST_3D` |
| **Spectral Field Evolver** | **SFEvolver** | Continue physical spatial-spectral evolution from the estimated field | second/subsequent `WaveMST_3D` |
| **Spectral Wave Propagator** | **SWP** | The damped-wave closed-form propagation unit used inside SFE/SFEvolver blocks | `WPO3D` / `WPO3DBlock` |
| **Mask A** | — | CASSI mask-gated wave-field initialization inside SWP | `MaskGateA` |

The current paper structure is:

```text
CASSI Prep  ->  Spectral Field Estimator  ->  Spectral Field Evolver  ->  Reconstructed HSI
```

Do **not** describe this as deep unfolding, A-HQS, iterative optimization, or a stage-wise solver.

## 2. Symbols

| Symbol | Meaning | Definition / note |
|---|---|---|
| \(X\) | ground-truth HSI | \(X\in\mathbb{R}^{H\times W\times L}\), \(L=28\) |
| \(\hat X\) | reconstructed HSI | final output, \(\hat X=U_2\) for the current two-step model |
| \(Y\) | 2D CASSI measurement | spectral summation after mask modulation and dispersion |
| \(M\) | unshifted 3D coded mask | \(M\in\mathbb{R}^{H\times W\times L}\) |
| \(\Phi_s\) | shifted sensing mask | the dispersed/shifted mask used by forward and SWP gating |
| \(P\) | normalization map | \(P=\Phi\Phi^\top=\sum_\lambda \Phi_{s,\lambda}^2\) |
| \(H_0\) | dirty shift-back field | \(H_0=\Phi^\top(Y/P)\) |
| \(R_0\) | measurement residual | \(R_0=Y-\Phi H_0\) |
| \(Q_0\) | residual-adjoint cue | \(Q_0=\Phi^\top(R_0/P)\) |
| \(C_0\) | SFE input tensor | \(C_0=[H_0,Q_0,M,H_0\odot M]\) |
| \(Z_0\) | SFE-corrected initial field | \(Z_0=H_0+\Delta_\psi(C_0)\) |
| \(U_1\) | estimate-step output | output of SFE |
| \(U_2\) | evolve-step output | output of SFEvolver, also \(\hat X\) |
| \(\oplus\) | residual addition | figure symbol |
| \(\odot\) | element-wise multiplication | figure symbol |
| \(\mathrm{Conv}^{0}_{3\times3}\) | zero-initialized 3×3 convolution | keeps \(Z_0=H_0\) at initialization |

## 3. Active model family

| Model label | Configuration | Purpose |
|---|---|---|
| **SMILE-S** | 222 shared + SFE | parameter-efficient small model |
| **SMILE-M** | 222 non-shared + SFE | current core model; step-specific estimate/evolve |
| **SMILE-L** | 244 non-shared + SFE | larger version for peak PSNR |

All current ablations should use **222** as the clean baseline unless explicitly stated otherwise.

## 4. SWP wording

Use:

- **3D damped wave equation**
- **Fourier-domain closed-form propagation**
- **spectral-mode-dependent damping and propagation speed**
- **CASSI mask-gated wave field**

Avoid:

- wavelength-specific physical coefficients
- attention in Fourier disguise
- standalone historical side modules
- deep unfolding stage
- data-consistency iteration

## 5. Historical / non-current terms

All historical routes that are not part of the active paper method are collected in `legacy.md`. Do not enumerate them in the Method section, figure labels, or current ablation plan.

## 6. One-paragraph canonical description

SMILE² first converts a CASSI measurement \(Y\) and mask \(M\) into a dirty shift-back field \(H_0\) and a residual-adjoint cue \(Q_0\). A Spectral Field Estimator uses \([H_0,Q_0,M,H_0\odot M]\) to form a cleaner initial spectral field and produces \(U_1\). A Spectral Field Evolver then applies SWP-based spatial-spectral physical evolution to obtain \(\hat X=U_2\). Inside SWP, Mask A gates the wave field with \(\Phi_s\), then a damped-wave closed-form kernel propagates it in the Fourier domain.


