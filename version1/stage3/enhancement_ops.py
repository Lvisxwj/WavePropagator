"""
enhancement_ops.py — Stage 2 物理增强模块

包含：
  - DispersionCorrector：空间色散 Born 一阶修正（模块 C）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DispersionCorrector(nn.Module):
    """空间色散 Born 一阶修正。

    预测空间依赖的折射率偏差 δv(r)，用 Laplacian(f) 做修正：
        f_out = f + weight * δv(r) * Laplacian(f)

    物理含义：不同空间位置有不同波传播速度（色散介质）。
    """

    def __init__(self, dim=28):
        super().__init__()
        # 预测 δv(r)：空间依赖的折射率偏差
        self.delta_v_net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # DWConv
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 1, bias=True),   # 压缩到单通道
            nn.Tanh(),                          # 限制范围 [-1, 1]
        )
        # 可学习修正强度
        self.weight = nn.Parameter(torch.tensor(0.1))

        # 固定 Laplacian 卷积核（不参与训练）
        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.register_buffer('laplacian_kernel', kernel)
        self.dim = dim

    def forward(self, f):
        """
        f: [B, C, H, W]
        返回: [B, C, H, W] 修正后的 f
        """
        # Laplacian（固定卷积，reflect padding 避免边界伪影）
        f_pad = F.pad(f, [1, 1, 1, 1], mode='reflect')
        laplacian_f = F.conv2d(f_pad, self.laplacian_kernel, groups=self.dim)

        # 空间依赖折射率偏差
        delta_v = self.delta_v_net(f)  # [B, 1, H, W]

        # Born 修正
        correction = delta_v * laplacian_f  # 广播 [B,1,H,W] * [B,C,H,W]
        return f + self.weight * correction
