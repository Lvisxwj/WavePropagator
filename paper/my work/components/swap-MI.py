"""
swap-MI.py — MI (Modulated Initialization)
对应代码：version2/model/mask_ops.py::MaskGateA
对应公式：(1.12) 软门控；(1.14) 卷积定理保持闭式解结构
颜色：背景 #f3f2f7（Part I），算子按 §1.2 标准
"""

import torch
import torch.nn as nn


class MI(nn.Module):
    """
    Inputs:
        x            : [B, Λ, H, W]
        mask_spatial : [B, Λ, H, W]  ∈ [0, 1]
    Outputs:
        u0, v0       : [B, Λ, H, W]  initial amplitude / velocity fields
    """

    def __init__(self, dim=28, eps=0.1):
        super().__init__()
        self.eps = eps
        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # DW 3x3
            nn.Conv2d(dim, dim, 1, bias=False),                    # 1x1
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x, mask_spatial):
        gate = self.eps + (1.0 - self.eps) * mask_spatial   # (1.12)
        u0 = self.phi(x) * gate
        v0 = self.psi(x) * gate
        return u0, v0
