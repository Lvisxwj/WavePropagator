"""Single-pass SMILE² E2E wrapper.

The backbone remains SWAP/WaveMST.  Optional physics-fused mode combines the
standard shift-back field H with a normalized adjoint field, then applies one
learnable data-consistency correction after the backbone.  There is no stage
recursion, LDE, rho, delta-Phi, or unfolding state.
"""

import math

import torch
import torch.nn as nn

from model.wpo3d import WaveMST_3D
from model.candidates import MaskConditionedInitialField, WavelengthAxisReconstruction


class SpectralLowRankResidual(nn.Module):
    """Scene-adaptive rank-r residual on the explicit 28-band output space."""

    def __init__(self, bands=28, rank=6, gamma_init=0.0):
        super().__init__()
        self.bands = int(bands)
        self.rank = int(rank)
        self.basis = nn.Linear(self.bands, self.bands * self.rank, bias=True)
        self.coefficients = nn.Sequential(
            nn.Conv2d(self.bands, self.bands, 3, 1, 1, groups=self.bands, bias=False),
            nn.GELU(),
            nn.Conv2d(self.bands, self.rank, 1, bias=True),
        )
        # Exact Base checkpoint equivalence at initialization.
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        pooled = x.mean(dim=(-2, -1))
        basis = self.basis(pooled).view(x.shape[0], self.bands, self.rank)
        basis = torch.nn.functional.normalize(basis, p=2, dim=1, eps=1e-6)
        coefficients = self.coefficients(x)
        residual = torch.einsum("bcr,brhw->bchw", basis, coefficients)
        return x + torch.tanh(self.gamma_raw) * residual


class MeasurementAwareInitialStateStem(nn.Module):
    """Zero-init measurement-aware residual stem for the first progressive step.

    It exposes the dirty shift-back field, a residual adjoint cue, the mask, and
    their direct interaction to the first full backbone.  The final projection is
    zero-initialized, so enabling the stem starts exactly from the original H0
    pathway and only learns if the cue is useful.
    """

    def __init__(self, bands=28, hidden=None):
        super().__init__()
        hidden = int(hidden or bands)
        self.net = nn.Sequential(
            nn.Conv2d(bands * 4, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, bands, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_field, residual_adjoint, mask):
        cue = torch.cat([h_field, residual_adjoint, mask, h_field * mask], dim=1)
        return h_field + self.net(cue)


def spectral_derivative_cues(x):
    """Return first/second absolute spectral-difference cues with x's shape.

    The cues are computed on the explicit band axis before SWAP.  They are not
    hard regularizers; they only expose local spectral continuity/curvature to
    the Step-1 estimator gate.
    """
    d1 = x.new_zeros(x.shape)
    d1[:, 1:] = x[:, 1:] - x[:, :-1]
    d1[:, 0] = d1[:, 1]

    d2 = x.new_zeros(x.shape)
    d2[:, 1:-1] = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    d2[:, 0] = d2[:, 1]
    d2[:, -1] = d2[:, -2]
    return d1.abs(), d2.abs()


class GatedSpectralFieldEstimateStep(nn.Module):
    """Step-1 gated clean-field estimator.

    This module is intentionally identity-preserving at initialization:
    the final correction projection is zero-initialized, so the initial output
    is exactly the dirty shift-back field.  The gate and estimator can only
    change the field after training finds the cues useful.
    """

    def __init__(self, bands=28, hidden=None):
        super().__init__()
        hidden = int(hidden or bands)
        cue_channels = bands * 6
        self.shared = nn.Sequential(
            nn.Conv2d(cue_channels, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
        )
        self.delta = nn.Conv2d(hidden, bands, 1, bias=True)
        self.gate = nn.Conv2d(hidden, bands, 1, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h_field, residual_adjoint, mask):
        d1, d2 = spectral_derivative_cues(h_field)
        cue = torch.cat([h_field, residual_adjoint, mask, h_field * mask, d1, d2], dim=1)
        feat = self.shared(cue)
        delta = self.delta(feat)
        gate = torch.sigmoid(self.gate(feat))
        return h_field + gate * delta


def shift_cube(x, step=2):
    """Shift [B,C,H,W] into CASSI coordinates [B,C,H,W+(C-1)step]."""
    b, c, h, w = x.shape
    out = x.new_zeros(b, c, h, w + (c - 1) * step)
    for band in range(c):
        out[:, band, :, band * step:band * step + w] = x[:, band]
    return out


def shift_back(y, bands=28, step=2):
    """Shift-back [B,H,W'] measurement to [B,C,H,W]."""
    b, h, wp = y.shape
    w = wp - (bands - 1) * step
    out = y.new_zeros(b, bands, h, w)
    for band in range(bands):
        out[:, band] = y[:, :, band * step:band * step + w]
    return out


def cassi_measure(x, mask, step=2):
    return shift_cube(x * mask, step=step).sum(dim=1)


def phi_phi_t(mask, step=2):
    return shift_cube(mask, step=step).square().sum(dim=1).clamp_min(1e-6)


def cassi_adjoint(q, mask, step=2):
    """Adjoint A^T(q), q [B,H,W'], mask [B,C,H,W]."""
    b, c, h, w = mask.shape
    out = mask.new_zeros(b, c, h, w)
    for band in range(c):
        out[:, band] = q[:, :, band * step:band * step + w] * mask[:, band]
    return out


class E2ESMILE(nn.Module):
    def __init__(
        self,
        dim=28,
        unet_stage=2,
        num_blocks=(2, 2, 2),
        use_sicmb=True,
        use_perchannel=True,
        use_spectral_wave=True,
        post_block="ffn",
        ffn_mult=4,
        input_mode="H",
        output_dc=False,
        dc_gamma_init=0.30,
        wpo_variant="full",
        gradient_checkpointing=False,
        step=2,
        bands=28,
        low_rank_residual_rank=0,
        low_rank_gamma_init=0.0,
        input_adapter="none",
        wavelength_cutoff_init=0.28,
        wave_param_mode="free",
        wave_basis_count=3,
        progressive_steps=1,
        progressive_share=True,
        return_intermediates=False,
        progressive_role_mode="plain",
        disable_sfevolver=False,
        use_mask_gate=True,
    ):
        super().__init__()
        if input_mode not in ("H", "dual_h_gap"):
            raise ValueError("input_mode must be H or dual_h_gap")
        self.input_mode = input_mode
        self.output_dc = bool(output_dc)
        self.step = int(step)
        self.bands = int(bands)
        self.low_rank_residual_rank = int(low_rank_residual_rank)
        self.input_adapter_name = str(input_adapter).lower()
        self.progressive_steps = int(progressive_steps)
        self.progressive_share = bool(progressive_share)
        self.return_intermediates = bool(return_intermediates)
        self.progressive_role_mode = str(progressive_role_mode).lower()
        self.disable_sfevolver = bool(disable_sfevolver)
        if self.progressive_steps < 1:
            raise ValueError("progressive_steps must be >= 1")
        if self.input_adapter_name not in ("none", "mask", "wavelength"):
            raise ValueError("input_adapter must be none/mask/wavelength")
        if self.progressive_role_mode not in ("plain", "estimate_evolve_v2", "gated_estimate_v1"):
            raise ValueError("progressive_role_mode must be plain, estimate_evolve_v2, or gated_estimate_v1")

        if input_mode == "dual_h_gap":
            self.input_fusion = nn.Conv2d(bands * 2, bands, 3, 1, 1, bias=False)
            # Start exactly from H; the physical branch is learned only if useful.
            nn.init.zeros_(self.input_fusion.weight)

        self.input_adapter = None
        if self.input_adapter_name == "mask":
            self.input_adapter = MaskConditionedInitialField(bands=bands)
        elif self.input_adapter_name == "wavelength":
            self.input_adapter = WavelengthAxisReconstruction(
                bands=bands, cutoff_init=wavelength_cutoff_init
            )

        self.initial_state_stem = None
        self.gated_estimate_step = None
        self.estimate_step_log_beta = None
        if self.progressive_role_mode == "estimate_evolve_v2":
            if self.progressive_steps < 2:
                raise ValueError("estimate_evolve_v2 requires progressive_steps>=2")
            self.initial_state_stem = MeasurementAwareInitialStateStem(bands=bands)
        elif self.progressive_role_mode == "gated_estimate_v1":
            if self.progressive_steps < 2 or self.progressive_share:
                raise ValueError("gated_estimate_v1 requires progressive_steps>=2 and progressive_share=False")
            self.gated_estimate_step = GatedSpectralFieldEstimateStep(bands=bands)
            # Identity-preserving partial-SWAP coefficient: exp(0)=1.
            # This is learnable instead of a hand-picked fixed partial strength.
            self.estimate_step_log_beta = nn.Parameter(torch.zeros(()))

        def make_backbone():
            return WaveMST_3D(
                dim=dim,
                stage=unet_stage,
                num_blocks=list(num_blocks),
                fbgw_mode="none",
                use_sicmb=use_sicmb,
                use_perchannel=use_perchannel,
                use_spectral_wave=use_spectral_wave,
                use_mask_gate=use_mask_gate,
                post_block=post_block,
                ffn_mult=ffn_mult,
                wpo_variant=wpo_variant,
                gradient_checkpointing=gradient_checkpointing,
                wave_param_mode=wave_param_mode,
                wave_basis_count=wave_basis_count,
            )

        if self.progressive_share:
            self.backbone = make_backbone()
            self.backbones = None
        else:
            self.backbone = None
            self.backbones = nn.ModuleList([make_backbone() for _ in range(self.progressive_steps)])

        self.low_rank_head = None
        if self.low_rank_residual_rank > 0:
            self.low_rank_head = SpectralLowRankResidual(
                bands=self.bands,
                rank=self.low_rank_residual_rank,
                gamma_init=low_rank_gamma_init,
            )

        if self.output_dc:
            gamma = min(max(float(dc_gamma_init), 1e-4), 1.0 - 1e-4)
            self.dc_logit = nn.Parameter(torch.tensor(math.log(gamma / (1.0 - gamma))))

    def forward(self, y, mask, shifted_mask=None, ppt=None):
        if ppt is None:
            ppt = phi_phi_t(mask, self.step)
        if shifted_mask is None:
            shifted_mask = shift_cube(mask, self.step)

        h_field = shift_back(y / self.bands * 2.0, self.bands, self.step)
        if self.input_mode == "dual_h_gap":
            gap_field = cassi_adjoint(y / ppt, mask, self.step)
            x0 = h_field + self.input_fusion(torch.cat([h_field, gap_field], dim=1))
        else:
            x0 = h_field

        if self.input_adapter is not None:
            x0 = self.input_adapter(x0, mask)
        if self.initial_state_stem is not None or self.gated_estimate_step is not None:
            residual0 = y - cassi_measure(h_field, mask, self.step)
            residual_adjoint = cassi_adjoint(residual0 / ppt, mask, self.step)
        if self.initial_state_stem is not None:
            x0 = self.initial_state_stem(x0, residual_adjoint, mask)
        if self.gated_estimate_step is not None:
            x0 = self.gated_estimate_step(x0, residual_adjoint, mask)

        if self.disable_sfevolver:
            return [x0] if self.return_intermediates else x0

        pred = x0
        outputs = []
        for step_idx in range(self.progressive_steps):
            if self.progressive_share:
                pred = self.backbone(pred, shifted_mask)
            else:
                next_pred = self.backbones[step_idx](pred, shifted_mask)
                if step_idx == 0 and self.progressive_role_mode == "gated_estimate_v1":
                    beta = torch.exp(self.estimate_step_log_beta)
                    pred = pred + beta * (next_pred - pred)
                else:
                    pred = next_pred
            outputs.append(pred)

        pred = outputs[-1]
        if self.low_rank_head is not None:
            pred = self.low_rank_head(pred)
            outputs[-1] = pred
        if self.output_dc:
            residual = y - cassi_measure(pred, mask, self.step)
            correction = cassi_adjoint(residual / ppt, mask, self.step)
            gamma = torch.sigmoid(self.dc_logit)
            pred = pred + gamma * correction
            outputs[-1] = pred
        if self.return_intermediates or self.progressive_steps > 1:
            return outputs
        return pred

    @property
    def dc_gamma(self):
        if not self.output_dc:
            return None
        return torch.sigmoid(self.dc_logit)

    @property
    def estimate_step_beta(self):
        if self.estimate_step_log_beta is None:
            return None
        return torch.exp(self.estimate_step_log_beta)


