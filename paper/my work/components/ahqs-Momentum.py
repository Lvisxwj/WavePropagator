"""
ahqs-Momentum.py — Nesterov Momentum (β_k)
对应代码：version2/model/unfolding.py::WPO_Unfold (use_ahqs=True 分支)
对应公式：(1.33) f̂ = f + β_k (f - f_prev), β_k = sigmoid(θ_k) ∈ (0, 1)
颜色：背景 #e6f1ff（Part III）
"""

import torch
import torch.nn as nn


class NesterovMomentum(nn.Module):
    """Inputs: f, f_prev ∈ [B,Λ,H,W] → Output: f_hat ∈ [B,Λ,H,W]."""

    def __init__(self, init_theta=0.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(init_theta))

    def forward(self, f, f_prev):
        beta = torch.sigmoid(self.theta)
        return f + beta * (f - f_prev)
