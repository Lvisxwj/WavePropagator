"""
refinement.py — 轻量局部精化模块

WPO 做全局传播后，补充局部纹理细节。
DWConv 3x3 + GELU + Conv 1x1，约 4.7K 参数（dim=28）。
"""

import torch.nn as nn


class LocalRefinement(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        hidden = dim * expand
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)
