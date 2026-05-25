"""
lde-SEC.py — SEC (Sensing Error Correction)
对应代码：version2/model/degradation.py::DegradationEstimation.delta_phi
对应公式：(1.28) ΔΦ = W2·LReLU(W1·Φ)
颜色：背景 #fff7e6（Part II）
"""

import torch.nn as nn


class SEC(nn.Module):
    """Inputs: Phi[B,Λ,H,W] → Output: delta_Phi[B,Λ,H,W]."""

    def __init__(self, dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, Phi):
        return self.net(Phi)
