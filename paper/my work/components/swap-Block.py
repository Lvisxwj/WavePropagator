"""
swap-Block.py — SWAP Block (LN + WPO3D + Res + LN + FFN + Res)
对应代码：version2/model/wpo3d.py::WPO3DBlock + WPO3D + FFN
对应公式：(1.9) WPO 主体；(1.39) 块组合
颜色：背景 #f3f2f7（Part I）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class WPO3D(nn.Module):
    """核心闭式解 wave-modulate；详见 algorithm.md (1.9)."""

    def __init__(self, dim, mask_mode='A'):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.vs    = nn.Parameter(torch.tensor(1.0))
        self.vl    = nn.Parameter(torch.tensor(0.5))
        self.t     = nn.Parameter(torch.tensor(1.0))
        self._lambda_sigma = nn.Parameter(torch.tensor(-2.0))
        self.out_norm   = LayerNorm2d(dim)
        self.out_linear = nn.Conv2d(dim, dim, 1, bias=False)
        # MI / KGD / AdaSpec 详见同目录 swap-MI / swap-AdaSpec / swap-KGD

    def forward(self, x, mask_spatial, sigma=None):
        # 1. α_eff = α + λ_σ σ  (1.36)
        # 2. MI: u0, v0
        # 3. 3D rFFT
        # 4. Wave modulate (Cs, Sn, decay)  (1.9)
        # 5. AdaSpec  (1.16)
        # 6. 3D irFFT
        # 7. [opt] KGD correction  (1.25)
        # 8. LN + SiLU(x) gate + Conv1x1
        raise NotImplementedError("详见 version2/model/wpo3d.py::WPO3D")


class SWAPBlock(nn.Module):
    def __init__(self, dim, mask_mode='A'):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo   = WPO3D(dim, mask_mode=mask_mode)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = FFN(dim)

    def forward(self, x, mask_spatial, sigma=None):
        x = x + self.wpo(self.norm1(x), mask_spatial, sigma=sigma)
        x = x + self.ffn(self.norm2(x))
        return x
