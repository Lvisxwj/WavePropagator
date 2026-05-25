"""
ahqs-ParaEstimator.py — Para Estimator (ρ_k)
对应代码：version2/model/utils.py::ParaEstimator
对应公式：(1.35) 中的步长 ρ_k = softplus(MLP(f))
颜色：背景 #e6f1ff（Part III）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParaEstimator(nn.Module):
    """Inputs: f ∈ [B,Λ,H,W] → Output: ρ_k ∈ [B,1,1,1] (positive)."""

    def __init__(self, in_nc=28, channel=32):
        super().__init__()
        self.fusion = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.bias = nn.Parameter(torch.FloatTensor([1.0]))
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1, padding=0, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fusion(x))
        x = self.avpool(x)
        x = self.mlp(x) + self.bias
        return F.softplus(x)
