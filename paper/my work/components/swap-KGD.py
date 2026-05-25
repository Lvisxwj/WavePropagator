"""
swap-KGD.py — KGD (Klein-Gordon Dispersion，可选)
对应代码：version2/model/mask_ops.py::MaskKleinGordonD
对应公式：(1.19)–(1.25)，Born 一阶修正
颜色：背景 #f3f2f7（Part I）
"""

import torch
import torch.nn as nn


class KGD(nn.Module):
    """
    Inputs:
        x            : [B, Λ, H, W]   (与 MI 相同的初始特征)
        mask_spatial : [B, Λ, H, W]
    Outputs:
        u0, v0       : [B, Λ, H, W]   (同 MI)
        m_sq         : [B, Λ, H, W]   质量场 m²(x,y) = m0² (1 - mask)
    """

    def __init__(self, dim=28, eps=0.1):
        super().__init__()
        self.eps = eps
        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.m0_sq = nn.Parameter(torch.tensor(0.1))     # (1.20)
        self.kg_weight = nn.Parameter(torch.tensor(0.1)) # w_KG

    def forward(self, x, mask_spatial):
        gate = self.eps + (1.0 - self.eps) * mask_spatial
        u0 = self.phi(x) * gate
        v0 = self.psi(x) * gate
        m_sq = self.m0_sq.clamp(0.0, 0.5) * (1.0 - mask_spatial)  # (1.20)
        return u0, v0, m_sq

    def apply_correction(self, u0_out, m_sq, sinc_term, decay, C, H, W):
        """Born 一阶修正 — (1.25)."""
        source = -m_sq * u0_out
        source_fft = torch.fft.rfftn(source, dim=(-3, -2, -1))
        corr_fft = source_fft * sinc_term * decay     # G(ω,t)
        correction = torch.fft.irfftn(corr_fft, s=(C, H, W), dim=(-3, -2, -1))
        return u0_out + self.kg_weight * correction
