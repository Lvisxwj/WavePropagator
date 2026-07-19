"""
cmb.py — Channel Modulation Block + Spatial Alignment Block

CMB 参考 SSR (CVPR 2024) Model.py:164-189, LRDUN lrdun.py:144-169。
  out = to_out(to_a(x) * to_v(x))
  参数量 ≈ 3*dim² + k²*dim

SAB 参考 SSR (CVPR 2024) Model.py。
  单通道空间注意力 × 逐通道空间权重。
  参数量 ≈ dim² + dim×9 + dim ≈ 1K。
"""

import torch.nn as nn


class CMB(nn.Module):
    def __init__(self, dim, k=11):
        super().__init__()
        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.to_out(self.to_a(x) * self.to_v(x))


class SAB(nn.Module):
    """Spatial Alignment Block — 空间对齐注意力。

    单通道空间 attention map × 逐通道空间权重，极轻量。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.Estimator = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1, bias=False), nn.GELU())
        self.SW = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.Sigmoid())
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, f):
        f = self.conv(f)
        out = self.SW(f) * self.Estimator(f).repeat(1, self.dim, 1, 1)
        return self.out(out)


class SimpleMix(nn.Module):
    """极简通道混合：Conv1×1 → GELU → Conv1×1（参照 DPU MPMLP）

    参数量：2 × dim² = 1568（dim=28）
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


