# SMILE² Logic Link

> Current authority: 2026-07-16  
> This file is the paper narrative chain. Historical branches are moved to `legacy.md`.


# 这个其实更像为什么设计sfe 和sfevolver的逻辑

## 1. Observation: CASSI starts from a dirty spectral field

The shift-back field \(H_0\) contains useful information but is contaminated by mask modulation, dispersion overlap, and spectral aliasing. The residual-adjoint cue \(Q_0=\Phi^\top((Y-\Phi H_0)/P)\) exposes what the current dirty field fails to explain under the CASSI forward model.

Therefore the first problem is not “run a generic image backbone,” but:

> estimate a cleaner initial spectral field from a dirty CASSI observation.

This motivates the **Spectral Field Estimator**.

## 2. Observation: HSI reconstruction needs structured spectral-spatial propagation

HSI bands are correlated but not identical. Neighboring bands often share structure, while material absorption and sensor response can create nonlocal or frequency-specific spectral behavior.

Attention can learn broad interactions, but it does not explicitly encode how information propagates over a 3D spatial-spectral field.

This motivates **SWP**, which models spatial-spectral propagation with a damped wave equation and a Fourier-domain closed-form solution.

## 3. Main idea

SMILE² reformulates E2E CASSI reconstruction as:

```text
Dirty CASSI field  ->  Spectral Field Estimation  ->  Physical Spectral Field Evolution
```

Formally:

$$
C_0=[H_0,Q_0,M,H_0\odot M],
\qquad
U_1=\mathcal{E}_{\theta_E}^{\mathrm{SWP}}(H_0+\Delta_\psi(C_0),\Phi_s),
$$

$$
\hat X=U_2=\mathcal{E}_{\theta_V}^{\mathrm{SWP}}(U_1,\Phi_s).
$$

## 4. Why two logical steps?

The two steps answer different questions:

- **SFE**: how to obtain a cleaner spectral field from a dirty CASSI initialization.
- **SFEvolver**: how to continue structured physical evolution after the field has been estimated.

This is not a deep unfolding split. There is no repeated measurement residual update between the two steps. The distinction is architectural and semantic: estimate first, evolve next.

## 5. Why SWP instead of pure token mixing?

SWP offers a structured global mixing operator:

1. Mask A gates the wave field with CASSI sensing structure.
2. 3D FFT maps the field into spatial-spectral frequency modes.
3. A damped-wave closed-form kernel controls propagation.
4. iFFT returns the evolved spatial-spectral field.

This provides an interpretable operator-level alternative to unconstrained pairwise attention.

## 6. Complexity statement

At the global-mixing operator level:

$$
\mathrm{Attention}:O(N^2),
\qquad
\mathrm{SWP}:O(N\log N).
$$

This should be written carefully: the claim is about the global mixing operator, not a guarantee that every full implementation is faster in every hardware setting.

## 7. Evidence to prioritize

Current paper evidence should focus on:

- Full model vs no-SFE baseline.
- Shared vs non-shared two-step evolution.
- \(U_1\) vs \(U_2\) intermediate outputs.
- Simulation metrics.
- Real-data qualitative results.
- Mechanism figures: CASSI Prep, SFE/SFEvolver signal flow, SWP closed-form propagation.

The most valuable explanation is:

> SFE cleans the dirty initial spectral field; SFEvolver performs structured spatial-spectral evolution.

## 8. What to avoid

Do not dilute the story with historical branches. They are archived in `legacy.md` and should not appear in the Method, main figures, or current ablation plan.

## 9. Final narrative in one paragraph

SMILE² treats CASSI reconstruction as learned spectral-field estimation followed by physical evolution. From the measurement and mask, CASSI Prep constructs a dirty shift-back field and a residual-adjoint cue. SFE estimates a cleaner spectral field, while SFEvolver propagates it through SWP, a mask-gated Fourier-domain damped-wave operator. This gives the network an interpretable spatial-spectral evolution bias while preserving a single-pass E2E reconstruction pipeline.

