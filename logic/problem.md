# SMILE² Problem Definition

> Current authority: 2026-07-16  
> This file defines the current paper problem. Historical explorations are in `legacy.md`.

## 1. Problem

CASSI reconstructs a 3D hyperspectral image \(X\in\mathbb{R}^{H\times W\times L}\) from a single 2D coded measurement \(Y\). The forward process masks, disperses, and sums spectral bands:

$$
Y=\sum_{\lambda}\mathrm{Shift}_{\lambda}(M_\lambda\odot X_\lambda).
$$

This is severely underdetermined: one 2D observation must recover a spatial-spectral field.

## 2. Limitation of black-box spectral mixing

Transformer-style methods model broad spatial-spectral interactions through learned token mixing. This is powerful, but the interaction pattern is usually hard to interpret and does not explicitly encode a physical propagation model over the HSI field.

For a paper about a new reconstruction paradigm, the key question is not only whether a network is larger or deeper, but:

> Can CASSI reconstruction be formulated as estimating an initial spectral field and then evolving it through a structured spatial-spectral physical operator?

> 这里可以点出一些transformer的共性问题，比如On方的复杂度，困难的全局建模能力，需要各种trick，窗口，而傅里叶变换，频域什么的，可以做到较好的全局

## 3. Limitation of expensive iterative reconstruction

Deep unfolding and physics-guided iterative methods can be accurate, but they introduce repeated update steps, higher training/inference cost, and possible numerical instability. Our current paper does not use unfolding stages, A-HQS, learned degradation estimators, or explicit data-consistency iterations as the main method.

The target is a **single-pass E2E** framework with a physically structured operator.

> 可以在符合边界的情况下吹自己，首个用波动方程物理算子来建模HSI重构任务的方法，什么的

## 4. Our formulation

SMILE² decomposes CASSI reconstruction into:

1. **CASSI Field Preparation**: derive \(H_0\), \(Q_0\), \(M\), and \(\Phi_s\) from \(Y\) and the mask.
2. **Spectral Field Estimation**: estimate a cleaner spectral field \(U_1\) from the dirty shift-back field and residual-adjoint cue.
3. **Spectral Field Evolution**: evolve \(U_1\) into \(\hat X\) using SWP-based spatial-spectral physical propagation.

The central scientific claim is:

> HSI reconstruction benefits from structured spatial-spectral evolution, not only from unconstrained token mixing.

## 5. Current method boundaries

Current method:

- SFE stem + first SWP-based U-Net = **Spectral Field Estimator**.
- Second SWP-based U-Net = **Spectral Field Evolver**.
- SWP = mask-gated damped-wave closed-form propagation.
- Mask A = \((\epsilon+(1-\epsilon)\Phi_s)\odot z\).

Non-current historical routes are not part of this problem definition. They are archived in `legacy.md`.

## 6. Evaluation logic

The main result should be evaluated by:

- PSNR / SSIM / SAM on simulation data;
- qualitative reconstruction on simulation and real data;
- SFE vs no-SFE ablation;
- shared vs non-shared two-step evolution;
- intermediate \(U_1\) vs final \(U_2\) behavior;
- mechanism figures showing structured spectral-spatial propagation.

If PSNR is not absolute SOTA, the paper should emphasize:

- single-pass physical evolution;
- interpretable spatial-spectral propagation;
- compact model family;
- SAM / spectral fidelity if supported by final metrics;
- fair comparison against E2E baselines.

## 7. Do not claim

- Do not claim the model is an iterative solver.
- Do not claim the entire network is always faster than every Transformer.
- Do not claim \(\alpha/v_s\) are literal wavelength-specific physical coefficients.
- Do not claim historical negative explorations are part of the method.
- Do not claim model size alone proves the paradigm.

