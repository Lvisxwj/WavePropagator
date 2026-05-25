"""
lde-DAG.py — DAG (Degradation-Aware Gating)
对应代码：version2/model/degradation.py::DegradationEstimation.deg_weight
对应公式：(1.29) w = sigmoid(U2·LReLU(U1·[Φ‖Φ*]))
颜色：背景 #fff7e6（Part II）
"""

import torch
import torch.nn as nn


class DAG(nn.Module):
    """Inputs: Phi, Phi_star ∈ [B,Λ,H,W] → Output: w ∈ (0,1)^{B×Λ×H×W}."""

    def __init__(self, dim=28, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, Phi, Phi_star):
        return self.net(torch.cat([Phi, Phi_star], dim=1))
