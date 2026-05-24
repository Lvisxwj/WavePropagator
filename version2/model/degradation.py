"""
degradation.py — 三合一退化估计模块

同时输出：
  1. delta_Phi  [B, C, H, W]  — sensing error 修正（用于 GD step）
  2. deg_weight [B, C, H, W]  — 空间退化权重（净化 WPO 初始场）
  3. sigma      [B, 1, 1, 1]  — 噪声水平（控制 WPO 阻尼 α_eff = α + λσ）

参考：
  - DPU (CVPR 2024) 的 DPB：退化 mask → 1×1 Conv → Sigmoid → 权重
  - DERNN-LNLT (2024) 的 DEN：残差学习估计 sensing error + noise level
"""

import torch
import torch.nn as nn


def construct_degraded_mask(Phi, len_shift=2):
    """构造退化 mask Phi*：shift → compress → reverse。

    参考 DPU/Model.py 的 reverse 操作（MST initialization）。

    Phi: [B, C, H, W] spatial mask
    返回: [B, C, H, W] 退化 mask（包含 shift+compression 退化信息）
    """
    B, C, H, W = Phi.shape
    # shift each band
    shifted = torch.zeros(B, C, H, W + (C - 1) * len_shift,
                          device=Phi.device, dtype=Phi.dtype)
    for c in range(C):
        shifted[:, c, :, c * len_shift: c * len_shift + W] = Phi[:, c, :, :]

    # compress (sum along spectral dim) and broadcast back
    compressed = shifted.sum(dim=1, keepdim=True)  # [B, 1, H, W']

    # reverse: assign back to each band
    Phi_star = torch.zeros_like(Phi)
    for c in range(C):
        Phi_star[:, c, :, :] = compressed[:, 0, :, c * len_shift: c * len_shift + W]

    # normalize
    Phi_star = 2.0 * Phi_star / C

    return Phi_star


class DegradationEstimation(nn.Module):
    """三合一退化估计

    参数量（dim=28, hidden=32）：
      delta_phi:  2 × 28 × 28 = 1568
      deg_weight: 56 × 32 + 32 × 28 = 2688
      sigma_est:  28 × 32 + 32 × 1 = 928
      总计: ~5.2K 参数
    """

    def __init__(self, dim=28, hidden=32):
        super().__init__()

        # 1. Sensing error 估计（参考 DERNN-LNLT）
        self.delta_phi = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

        # 2. 退化空间权重（参考 DPU 的 DPB）
        self.deg_weight = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )

        # 3. 噪声水平估计
        self.sigma_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # sigma > 0
        )

    def forward(self, f, Phi, Phi_star):
        """
        f:        [B, C, H, W] 当前迭代估计
        Phi:      [B, C, H, W] spatial mask
        Phi_star: [B, C, H, W] 退化 mask（construct_degraded_mask 的输出）

        Returns:
            delta_Phi:  [B, C, H, W] sensing error
            deg_weight: [B, C, H, W] 退化权重（0~1）
            sigma:      [B, 1, 1, 1] 噪声水平
        """
        delta_Phi = self.delta_phi(Phi)
        deg_weight = self.deg_weight(torch.cat([Phi, Phi_star], dim=1))
        sigma = self.sigma_est(f).view(-1, 1, 1, 1)

        return delta_Phi, deg_weight, sigma
