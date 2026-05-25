"""
lde.py — Part II 顶层: LDE (Learned Degradation Estimator)
对应代码：version2/model/degradation.py::DegradationEstimation
对应公式：(1.28) SEC；(1.29) DAG；(1.30) NLE；(1.13.2) 退化 mask 构造
颜色：背景 #fff7e6（Part II）
"""

import torch
import torch.nn as nn


class LDE(nn.Module):
    """
    Inputs:
        f        : [B, Λ, H, W]  当前迭代估计
        Phi      : [B, Λ, H, W]  spatial mask
        Phi_star : [B, Λ, H, W]  退化 mask（预计算）
    Outputs:
        delta_Phi  : [B, Λ, H, W]   ↔ SEC, (1.28)
        deg_weight : [B, Λ, H, W]   ↔ DAG, (1.29)
        sigma      : [B, 1, 1, 1]   ↔ NLE, (1.30)
    """

    def __init__(self, dim=28, hidden=32):
        super().__init__()
        # SEC
        self.delta_phi = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        # DAG
        self.deg_weight = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        # NLE
        self.sigma_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, f, Phi, Phi_star):
        delta_Phi = self.delta_phi(Phi)                                # (1.28)
        deg_weight = self.deg_weight(torch.cat([Phi, Phi_star], dim=1)) # (1.29)
        sigma = self.sigma_est(f).view(-1, 1, 1, 1)                    # (1.30)
        return delta_Phi, deg_weight, sigma


def construct_degraded_mask(Phi, len_shift=2):
    """对应 (1.13.2)：shift → compress → reverse → normalize. 预计算一次."""
    B, C, H, W = Phi.shape
    shifted = torch.zeros(B, C, H, W + (C - 1) * len_shift,
                          device=Phi.device, dtype=Phi.dtype)
    for c in range(C):
        shifted[:, c, :, c * len_shift: c * len_shift + W] = Phi[:, c, :, :]
    compressed = shifted.sum(dim=1, keepdim=True)
    Phi_star = torch.zeros_like(Phi)
    for c in range(C):
        Phi_star[:, c, :, :] = compressed[:, 0, :, c * len_shift: c * len_shift + W]
    return 2.0 * Phi_star / C
