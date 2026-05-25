"""
ahqs.py — Part III 顶层: A-HQS (Accelerated Half-Quadratic Splitting)
对应代码：version2/model/unfolding.py::WPO_Unfold
对应公式：(1.32)–(1.37) 单 stage 完整流程；(1.38) 多 stage 损失
颜色：背景 #e6f1ff（Part III）
"""

import torch
import torch.nn as nn


class AHQS(nn.Module):
    """
    K-stage Accelerated HQS unfolding.

    Per-stage flow (see equation (1.37)):
        1. LDE: ΔΦ, w, σ = LDE(f^{k-1}, Φ, Φ*)
        2. Momentum:  f̂ = f + β_k (f - f_prev)
        3. GD step (closed form):  z = f̂ + ρ_k Φ_eff^T (g - Φ_eff f̂) / (μ + ΦΦ^T)
        4. Purify:    z_clean = z · (1 + w)
        5. SWAP:      f_wave = SWAP(z_clean, Φ, σ)   # σ → α_eff
        6. LRB:       f^k = f_wave + LRB(f_wave)
    """

    def __init__(self, dim=28, num_stages=5, share_weights=True, use_ahqs=False,
                 len_shift=2, size=256):
        super().__init__()
        self.num_stages = num_stages
        self.share_weights = share_weights
        self.use_ahqs = use_ahqs
        self.len_shift = len_shift
        self.size = size
        self.nC = dim

        # ParaEstimator each stage (always per-stage)
        from importlib import import_module
        ParaEstimator = import_module('ahqs-ParaEstimator').ParaEstimator
        self.rho = nn.ModuleList([ParaEstimator(in_nc=dim) for _ in range(num_stages)])

        # Nesterov beta per stage (sigmoid-parameterized)
        if use_ahqs:
            self.betas = nn.ParameterList(
                [nn.Parameter(torch.tensor(0.0)) for _ in range(num_stages)]
            )

        # LDE / SWAP / LRB references (shared or per stage)
        LDE = import_module('lde').LDE
        SWAP = import_module('swap').SWAP
        LRB = import_module('lde-LRB').LRB
        if share_weights:
            self.lde, self.swap, self.lrb = LDE(dim), SWAP(dim), LRB(dim)
        else:
            self.ldes  = nn.ModuleList([LDE(dim)  for _ in range(num_stages)])
            self.swaps = nn.ModuleList([SWAP(dim) for _ in range(num_stages)])
            self.lrbs  = nn.ModuleList([LRB(dim)  for _ in range(num_stages)])

        self.initial_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)

    def _stage_components(self, k):
        if self.share_weights:
            return self.lde, self.swap, self.lrb
        return self.ldes[k], self.swaps[k], self.lrbs[k]

    def forward(self, g, Phi, PhiPhiT, Phi_star):
        # Initialization (see (1.34) μ=0 baseline)
        g_normal = g / self.nC * 2
        # NOTE: shift_back / mul_Phi_f are in version2/model/utils.py
        from importlib import import_module
        u = import_module('version2.model.utils')
        f0 = u.shift_back_batch(g_normal.repeat(1, self.nC, 1, 1), self.len_shift, self.size)
        f = self.initial_conv(torch.cat([f0, Phi], dim=1))

        if self.use_ahqs:
            f_prev = f.clone()
        outputs = []
        Phi_shift = u.shift_batch(Phi, self.len_shift)

        for k in range(self.num_stages):
            lde, swap, lrb = self._stage_components(k)
            delta_Phi, w, sigma = lde(f, Phi, Phi_star)        # (1.28)–(1.30)

            if self.use_ahqs:
                beta = torch.sigmoid(self.betas[k])             # (1.33)
                f_hat = f + beta * (f - f_prev)
                f_prev = f.detach().clone()
                Phi_eff = u.shift_batch(Phi + delta_Phi, self.len_shift)
                rho = self.rho[k](f_hat)
                Phi_f = u.mul_Phi_f(Phi_eff, f_hat, self.len_shift)
                residual = ((g - Phi_f) / PhiPhiT.clamp(min=1e-6)).clamp(-10, 10)
                z = f_hat + rho * u.mul_PhiT_residual(Phi_eff, residual, self.len_shift, self.size)
            else:
                rho = self.rho[k](f)
                Phi_f = u.mul_Phi_f(Phi_shift, f, self.len_shift)
                residual = ((g - Phi_f) / PhiPhiT.clamp(min=1e-6)).clamp(-10, 10)
                z = f + rho * u.mul_PhiT_residual(Phi_shift, residual, self.len_shift, self.size)

            z_clean = z * (1.0 + w)                              # (1.36)
            f_wave  = swap(z_clean, Phi, sigma=sigma)             # (1.37)
            f       = f_wave + lrb(f_wave)                        # (1.31)+(1.37)
            outputs.append(f)
        return outputs
