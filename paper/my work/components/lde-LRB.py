"""
lde-LRB.py — LRB (Local Refinement Block)
对应代码：version2/model/refinement.py::LocalRefinement
对应公式：(1.31) LRB(x) = C3(DW3x3(GELU(C1·x)))
颜色：背景 #fff7e6（Part II），部署位置在 SWAP 之后
"""

import torch.nn as nn


class LRB(nn.Module):
    """Inputs: x ∈ [B,Λ,H,W] → Output: same shape (residual added externally)."""

    def __init__(self, dim=28, expand=2):
        super().__init__()
        h = dim * expand
        self.net = nn.Sequential(
            nn.Conv2d(dim, h, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(h, h, 3, 1, 1, groups=h, bias=False),
            nn.GELU(),
            nn.Conv2d(h, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)
