"""
lde-NLE.py — NLE (Noise Level Estimator)
对应代码：version2/model/degradation.py::DegradationEstimation.sigma_est
对应公式：(1.30) σ = softplus(V2·ReLU(V1·GAP(f)))
颜色：背景 #fff7e6（Part II）
"""

import torch.nn as nn


class NLE(nn.Module):
    """Inputs: f ∈ [B,Λ,H,W] → Output: σ ∈ [B,1,1,1]."""

    def __init__(self, dim=28, hidden=32):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, f):
        return self.mlp(self.gap(f)).view(-1, 1, 1, 1)
