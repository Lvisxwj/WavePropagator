# SMILE² Algorithm Specification

This document is the notation and implementation reference for the current SMILE² paper/code line. It is the source-of-truth for formulas, dimensions, signal flow, physical correspondence, and code mapping.

Current method:

$$
(Y,M) \xrightarrow{\text{CASSI preparation}} (H_0,Q_0,M,\Phi_s,P)
\xrightarrow{\text{SFEstimator}} U_1
\xrightarrow{\text{SFEvolver}} U_2=\widehat X .
$$

The core operator inside both field components is the Spectral Wave Propagator (SWP), a Fourier-domain closed-form solution of a learnable damped wave equation on a sampled spatial--spectral field.

---

## 1. Notation and dimensions

### 1.1 Field convention

An HSI cube is treated as a sampled spatial--spectral field:

$$
X\in\mathbb{R}^{L\times H\times W},
$$

where $L$ is the number of spectral bands and $(H,W)$ is the spatial size. In the TSA simulation benchmark, $L=28$. In implementation, tensors are batch-first:

$$
X\in\mathbb{R}^{B\times L\times H\times W}.
$$

The symbol $\lambda$ denotes the discrete spectral coordinate. The spectral FFT used by SWP is a structured basis over sampled bands; it is not a claim that real spectra are physically periodic.

Inside the U-Net backbone, the channel dimension may become a latent feature dimension $C$. The input/output field remains $L$-band HSI, while SWP applies the same spatial--spectral operator form to intermediate feature fields.

### 1.2 Main symbols

| Symbol | Code name | Shape | Meaning |
|---|---|---:|---|
| $X$ | `gt`, `x` | $B\times L\times H\times W$ | Ground-truth HSI. |
| $\widehat X$ | `pred`, `outputs[-1]` | $B\times L\times H\times W$ | Final reconstruction. |
| $Y$ | `y` | $B\times H\times W_s$ | 2D CASSI measurement. |
| $M$ | `mask` | $B\times L\times H\times W$ | Band-aligned mask cube. |
| $\mathcal S_\ell$ | `shift_cube` | -- | Horizontal dispersion shift for band $\ell$. |
| $W_s$ | shifted width | $W+d(L-1)$ | Measurement width after dispersion. |
| $\Phi$ | implicit operator | -- | CASSI sensing: mask, shift, sum. |
| $\Phi^\top$ | `cassi_adjoint` | -- | CASSI adjoint/back-projection. |
| $\Phi_s$ | `shifted_mask` | $B\times L\times H\times W_s$ | Shifted mask cue for mask-conditioned propagation. |
| $P$ | `ppt` | $B\times H\times W_s$ | Normalization map $\Phi\Phi^\top\mathbf 1$. |
| $H_0$ | `h_field` | $B\times L\times H\times W$ | Dirty shift-back spectral field. |
| $R_0$ | `residual0` | $B\times H\times W_s$ | Residual in measurement plane. |
| $Q_0$ | `residual_adjoint` | $B\times L\times H\times W$ | Normalized residual-adjoint cue. |
| $U_1$ | `outputs[0]` | $B\times L\times H\times W$ | Estimated cleaner spectral field. |
| $U_2$ | `outputs[-1]` | $B\times L\times H\times W$ | Evolved field and final reconstruction. |

Default CASSI shift interval: $d=2$. For $H=W=256,L=28$, $W_s=310$.

---

## 2. CASSI forward model

For each spectral band $\ell$, coded-aperture modulation gives

$$
Z_\ell=M_\ell\odot X_\ell.
$$

The dispersive element shifts this coded slice:

$$
\mathcal S_\ell(Z_\ell)(i,j+d(\ell-1))=Z_\ell(i,j).
$$

The detector integrates all shifted coded bands:

$$
Y=\Phi X=\sum_{\ell=1}^{L}\mathcal S_\ell(M_\ell\odot X_\ell),
\qquad Y\in\mathbb{R}^{B\times H\times W_s}.
$$

Code mapping:

```python
shifted = shift_cube(x * mask, step=2)  # [B,L,H,W_s]
y = shifted.sum(dim=1)                  # [B,H,W_s]
```

Implemented by `shift_cube` and `cassi_measure` in `release/src/repo/model/smile.py`.

---

## 3. CASSI preparation

CASSI preparation constructs measurement-consistent cues before neural field estimation/evolution.

### 3.1 CASSI adjoint $\Phi^\top$

For any detector-plane tensor $q\in\mathbb{R}^{B\times H\times W_s}$,

$$
\Phi^\top q=
\left[M_\ell\odot\mathcal S_\ell^{-1}(q)\right]_{\ell=1}^{L}
\in\mathbb{R}^{B\times L\times H\times W}.
$$

Code:

```python
out[:, band] = q[:, :, band * step: band * step + W] * mask[:, band]
```

Implemented by `cassi_adjoint(q, mask, step=2)`.

### 3.2 Normalization map $P=\Phi\Phi^\top\mathbf 1$

Overlapped shifted masks produce nonuniform measurement-plane coverage. The code uses

$$
P=\Phi\Phi^\top\mathbf 1
=\sum_{\ell=1}^{L}\mathcal S_\ell(M_\ell)^2
\in\mathbb{R}^{B\times H\times W_s}.
$$

For numerical safety,

$$
P\leftarrow\max(P,10^{-6}).
$$

Code:

```python
P = shift_cube(mask, step=2).square().sum(dim=1).clamp_min(1e-6)
```

Implemented by `phi_phi_t(mask, step=2)`.

### 3.3 Dirty shift-back field $H_0$

The first band-aligned field is a shift-back measurement cube:

$$
H_0=\operatorname{ShiftBack}\left(\frac{2}{L}Y\right)
\in\mathbb{R}^{B\times L\times H\times W}.
$$

$H_0$ is not clean HSI. It contains coded spectral overlap because each detector pixel receives contributions from multiple shifted bands.

Code:

```python
h_field = shift_back(y / bands * 2.0, bands, step)
```

### 3.4 Normalized residual-adjoint cue $Q_0$

Project $H_0$ back to the measurement plane:

$$
R_0=Y-\Phi H_0.
$$

Then compute the normalized adjoint:

$$
Q_0=\Phi^\top\left(\frac{R_0}{P}\right)
\in\mathbb{R}^{B\times L\times H\times W}.
$$

Code:

```python
residual0 = y - cassi_measure(h_field, mask, step)
Q0 = cassi_adjoint(residual0 / P, mask, step)
```

$Q_0$ exposes measurement inconsistency left by the dirty shift-back field. It is a one-shot preparation cue, not a recurrent optimization state.

### 3.5 Shifted mask cue $\Phi_s$

The shifted mask cube is

$$
\Phi_s=[\mathcal S_\ell(M_\ell)]_{\ell=1}^{L}
\in\mathbb{R}^{B\times L\times H\times W_s}.
$$

It is passed into the backbone as the mask-conditioning cue. At each U-Net resolution, the implementation aligns it to the feature resolution by cropping/downsampling, producing an effective mask cue

$$
M_r\in\mathbb{R}^{B\times C_r\times H_r\times W_r}.
$$

Code:

```python
shifted_mask = shift_cube(mask, step)
mask_spatial = input_mask[:, :, :, :H] if input_mask.shape[-1] > H else input_mask
mask_spatial = torch.sigmoid(mask_down(mask_spatial))
```

### 3.6 Prepared cue stack

The explicit cue stack for the estimator stem is

$$
C_0=[H_0,Q_0,M,H_0\odot M]
\in\mathbb{R}^{B\times 4L\times H\times W}.
$$

Code:

```python
cue = torch.cat([h_field, residual_adjoint, mask, h_field * mask], dim=1)
```

---

## 4. Overall signal flow

The method consists of two named field components:

$$
U_1=\operatorname{SFEstimator}(H_0,Q_0,M,\Phi_s),
$$

$$
U_2=\operatorname{SFEvolver}(U_1,\Phi_s),
$$

$$
\widehat X=U_2.
$$

SMILE-S/M/L are compact scaling variants:

- SMILE-S: `[2,2,2]`, estimator/evolver share backbone parameters.
- SMILE-M: `[2,2,2]`, estimator/evolver use separate backbone parameters.
- SMILE-L: `[2,4,4]`, separate estimator/evolver backbones with larger capacity.


---

## 5. SFEstimator

### 5.1 Purpose

SFEstimator transforms the dirty shift-back field into a cleaner initial spectral field by using measurement-consistent cues:

$$
(H_0,Q_0,M,H_0\odot M)\rightarrow U_1.
$$

### 5.2 Zero-initialized measurement-aware stem

The first correction is

$$
\Delta H_0=
\operatorname{Conv}^{0}_{3\times3}
\left(
\operatorname{GELU}(\operatorname{Conv}_{3\times3}(C_0))
\right),
$$

$$
Z_0=H_0+\Delta H_0.
$$

$\operatorname{Conv}^{0}_{3\times3}$ means the final convolution is zero-initialized. Therefore at initialization

$$
Z_0=H_0.
$$

This makes the estimator identity-preserving at the start of training and lets it learn only useful measurement-aware corrections.

Code: `MeasurementAwareInitialStateStem`.

### 5.3 Estimator backbone

The stem output enters a SWP-based U-Net backbone:

$$
U_1=\mathcal B_E(Z_0,\Phi_s).
$$

Each local block applies

$$
F\leftarrow F+\operatorname{SWP}(\operatorname{LN}(F),M_r),
$$

$$
F\leftarrow F+\operatorname{FFN}(\operatorname{LN}(F)).
$$

The output is still an explicit $L$-band spectral field.

---

## 6. SFEvolver

SFEvolver receives $U_1$ and produces the final reconstruction:

$$
U_2=\mathcal B_V(U_1,\Phi_s),
\qquad \widehat X=U_2.
$$

It uses the same SWP-based U-Net form:

1. `embedding`: $L$-band field to feature channels;
2. encoder: SWP blocks and downsampling;
3. bottleneck: SWP blocks;
4. decoder: upsampling, skip fusion, SWP blocks;
5. `mapping`: feature channels to $L$-band residual;
6. residual output:

$$
\mathcal B(F,\Phi_s)=F+\operatorname{Mapping}(\operatorname{Decoder}(\operatorname{Encoder}(F,\Phi_s))).
$$

Code: `SpectralFieldBackbone.forward`.

Inside each SWP operator, the propagated wave output is multiplied by a content-dependent gate:

$$
O=\operatorname{OutLinear}(\operatorname{LN}(u(t))\odot\Gamma(F_g)),
$$

where $\Gamma$ is implemented by `Conv1x1 + depth-wise Conv11x11`. This gate is part of the field evolution operator.

---

## 7. Spectral Wave Propagator (SWP)

SWP is the core spatial--spectral propagation operator.

### 7.1 Local projection

Given feature field $F\in\mathbb{R}^{B\times C\times H\times W}$, SWP first computes

$$
[F_w,F_g]=\operatorname{Proj}(\operatorname{DWConv}_{3\times3}(F)).
$$

$F_w$ is the wave branch. $F_g$ is the content branch for output modulation.

### 7.2 Mask-conditioned initial displacement and velocity

The effective mask gate is

$$
G(M_r)=\epsilon_m+(1-\epsilon_m)M_r,
\qquad \epsilon_m=0.1.
$$

Learnable maps generate initial displacement and velocity:

$$
u_0=\phi(F_w)\odot G(M_r),
$$

$$
v_0=\psi(F_w)\odot G(M_r).
$$

In code, $\phi$ and $\psi$ are depth-wise `3x3` plus point-wise `1x1` convolutions. This is `MaskConditionedGate`.

### 7.3 Damped wave equation

SWP adapts an anisotropic damped wave equation to the sampled spatial--spectral field:

$$
\partial_{tt}u+\alpha\partial_tu
=v_s^2(\partial_{xx}+\partial_{yy})u+v_\lambda^2\partial_{\lambda\lambda}u.
$$

Physical correspondence:

- $u$: propagated feature/spectral field;
- $t$: learnable evolution time;
- $\alpha$: damping coefficient;
- $v_s$: spatial propagation speed;
- $v_\lambda$: spectral propagation speed.

### 7.4 Fourier-domain ODE

Let

$$
\widehat u(\mathbf f,t)=\mathcal F_{x,y,\lambda}[u(x,y,\lambda,t)],
\qquad \mathbf f=(f_x,f_y,f_\lambda).
$$

Each frequency mode satisfies

$$
\partial_{tt}\widehat u
+\alpha(f_\lambda)\partial_t\widehat u
+\omega_0^2(\mathbf f)\widehat u=0,
$$

with

$$
\omega_0^2(\mathbf f)=(2\pi)^2
\left[v_s^2(f_\lambda)(f_x^2+f_y^2)+v_\lambda^2f_\lambda^2\right].
$$

The code enforces positive parameters by `softplus`:

$$
\alpha=\operatorname{softplus}(a),\quad
v_s=\operatorname{softplus}(s),\quad
v_\lambda=\operatorname{softplus}(l),\quad
t=\operatorname{softplus}(\tau).
$$

### 7.5 Closed-form solution

Define

$$
\eta=\omega_0^2-\left(\frac{\alpha}{2}\right)^2.
$$

Given $\widehat u_0$ and $\widehat v_0$,

$$
\widehat u(t)=e^{-\alpha t/2}
\left[
\widehat u_0 C_s(\eta,t)
+
\left(\widehat v_0+\frac{\alpha}{2}\widehat u_0\right)S_n(\eta,t)
\right].
$$

For $\eta\ge0$:

$$
C_s(\eta,t)=\cos(\sqrt\eta t),
\qquad
S_n(\eta,t)=\frac{\sin(\sqrt\eta t)}{\sqrt\eta}.
$$

For $\eta<0$, the code uses a fused exponential form equivalent to damped `cosh/sinh` to avoid overflow:

$$
e^{-\alpha t/2}C_s=\frac{e^{(\gamma-\alpha/2)t}+e^{-(\gamma+\alpha/2)t}}{2},
$$

$$
e^{-\alpha t/2}S_n=\frac{e^{(\gamma-\alpha/2)t}-e^{-(\gamma+\alpha/2)t}}{2\gamma},
\qquad \gamma=\sqrt{-\eta}.
$$

Implementation: `SpectralWavePropagator._solve_damped`.

### 7.6 FFT implementation

SWP solves the ODE globally in the Fourier domain:

$$
\widehat u_0=\mathcal F_{x,y,\lambda}(u_0),
\qquad
\widehat v_0=\mathcal F_{x,y,\lambda}(v_0),
$$

$$
u(t)=\mathcal F^{-1}_{x,y,\lambda}(\widehat u(t)).
$$

Code uses `torch.fft.rfftn` and `torch.fft.irfftn` along `(-3,-2,-1)`, i.e., channel/spectral, height, and width dimensions. The channel dimension can be padded to the next power of two for FFT efficiency and then cropped back.

### 7.7 Residual SWP block

Each block is

$$
F'=F+\operatorname{SWP}(\operatorname{LN}(F),M_r),
$$

$$
F^+=F'+\operatorname{FFN}(\operatorname{LN}(F')).
$$

Implementation: `SWPBlock.forward`.

---

## 8. Complexity

For a SWP feature field with

$$
N=CHW,
$$

its global mixing is dominated by 3D FFT and inverse FFT:

$$
\mathcal O(N\log N).
$$

This is an operator-level complexity statement. End-to-end runtime also depends on U-Net depth, channel width, hardware, memory traffic, and implementation.

---

## 9. Training objective

The current main configuration returns two fields $(U_1,U_2)$. Training uses weighted RMSE supervision:

$$
\mathcal L
=
\sum_{i=1}^{2}\beta_i
\sqrt{\operatorname{MSE}(U_i,X)+10^{-8}},
\qquad
\boldsymbol\beta=[0.2,1.0].
$$

No SAM loss, SSIM loss, or spectral-angle loss is used. SAM is reported only as an evaluation metric.

Implementation:

```python
loss = sum(w * rmse_loss(output, gt) for w, output in zip([0.2, 1.0], outputs))
```

---

## 10. Pseudocode

```text
Input: measurement Y, mask cube M
Output: reconstructed HSI X_hat

1. Phi_s = ShiftCube(M)
2. P     = sum_l ShiftCube(M_l)^2, clamped by eps
3. H0    = ShiftBack(2Y / L)
4. R0    = Y - Phi(H0)
5. Q0    = Phi^T(R0 / P)
6. C0    = concat(H0, Q0, M, H0 * M)

7. Z0    = H0 + ZeroInitConv(GELU(Conv(C0)))
8. U1    = SpectralFieldBackbone_E(Z0, Phi_s)   # SFEstimator
9. U2    = SpectralFieldBackbone_V(U1, Phi_s)   # SFEvolver
10. Xhat = U2
```

Inside SWP:

```text
1. Fw, Fg = Project(DWConv(LN(F)))
2. gate   = eps + (1 - eps) * aligned_mask
3. u0     = phi(Fw) * gate
4. v0     = psi(Fw) * gate
5. u0_hat, v0_hat = FFT3D(u0), FFT3D(v0)
6. solve damped-wave ODE for every frequency mode
7. u_t    = IFFT3D(u_hat(t))
8. out    = OutLinear(LN(u_t) * ContentGate(Fg))
9. F      = F + out
```

---

## 11. Code mapping

| Concept | Code location |
|---|---|
| CASSI shift | `release/src/repo/model/smile.py::shift_cube` |
| CASSI measurement $\Phi X$ | `release/src/repo/model/smile.py::cassi_measure` |
| $P=\Phi\Phi^\top\mathbf 1$ | `release/src/repo/model/smile.py::phi_phi_t` |
| CASSI adjoint $\Phi^\top q$ | `release/src/repo/model/smile.py::cassi_adjoint` |
| Dirty shift-back $H_0$ | `release/src/repo/model/smile.py::shift_back`, `SMILE2.forward` |
| Measurement-aware stem | `release/src/repo/model/smile.py::MeasurementAwareInitialStateStem` |
| SMILE² wrapper | `release/src/repo/model/smile.py::SMILE2` |
| Field backbone | `release/src/repo/model/spectral_wave_propagator.py::SpectralFieldBackbone` |
| SWP block | `release/src/repo/model/spectral_wave_propagator.py::SWPBlock` |
| SWP operator | `release/src/repo/model/spectral_wave_propagator.py::SpectralWavePropagator` |
| Mask-conditioned initial fields | `release/src/repo/model/cassi_operators.py::MaskConditionedGate` |
| Stable damped-wave solve | `release/src/repo/model/spectral_wave_propagator.py::_solve_damped` |
| RMSE loss | `release/src/repo/loss.py::rmse_loss` |
| Multi-field loss | `release/src/repo/train.py::multi_output_reconstruction_loss` |
| Configs | `release/src/repo/configs/smile_s.yaml`, `smile_m.yaml`, `smile_l.yaml` |

---

## 12. Fixed terminology

Use the following terms consistently:

- **SMILE²**: full method/framework.
- **CASSI preparation**: construction of $H_0,Q_0,M,\Phi_s,P$.
- **SFEstimator**: estimates $U_1$ from measurement-consistent cues.
- **SFEvolver**: evolves $U_1$ into $U_2=\widehat X$.
- **SWP / Spectral Wave Propagator**: Fourier-domain damped-wave propagation operator.
- **spectral-mode-dependent damping and propagation speed**: description of $\alpha(f_\lambda),v_s(f_\lambda),v_\lambda$.
- **mask-conditioned propagation**: SWP uses the aligned shifted-mask cue in initial displacement/velocity generation.


