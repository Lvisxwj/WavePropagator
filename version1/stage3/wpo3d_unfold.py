"""
wpo3d_unfold.py — Model 7/8: Deep Unfolding 版 WaveMST

K-stage GAP unfolding：
  每个 stage = GD step（数据保真）+ WPO3D prior（物理先验）

Model 7: WaveMST_3D_Unfold  — 3D-WPO Pure 的 unfolding 版
Model 8: WaveMST_KG_Unfold  — 3D-WPO-KG 的 unfolding 版

原始 WPO 模块（wpo3d.py）完全不修改，仅在外部包装 unfolding 循环。

物理增强模块（可选）：
  - 源项注入（use_source_injection）：Φ^T g 拼接到 prior 输入
  - 色散修正（use_dispersive）：空间依赖 Born 修正

参考：SSR/Model.py Net（GAP 风格 unfolding）
"""

import torch
import torch.nn as nn
from wpo3d import WaveMST_3D, WaveMST_KG
from unfolding_ops import (
    shift_batch, shift_back_batch,
    mul_Phi_f, mul_PhiT_residual,
    ParaEstimator,
)


class WaveMST_3D_Unfold(nn.Module):
    """3D-WPO 的 K-stage GAP unfolding 包装类。

    架构：K 个 stage，每个 stage = GD step + WPO3D prior

    Args:
        dim:            通道数（默认 28）
        stage:          U-Net 内部的 stage 数（传给 WaveMST_3D）
        num_blocks:     每层的 block 数
        num_stages:     unfolding 的 K（stage 数）
        share_weights:  True 时所有 stage 共享同一个 WPO prior
        use_kg:         True 时使用 KG 方程（Model 8）
        mask_mode:      传给 WPO prior
        size:           空间尺寸（crop_size，用于 shift_back）
        len_shift:      CASSI 位移步长（默认 2）
        use_source_injection: 模块 A — Φ^T g 源项注入
        use_dispersive:       模块 C — 空间色散修正
    """

    def __init__(self, dim=28, stage=2, num_blocks=None,
                 num_stages=5, share_weights=False, use_kg=False,
                 mask_mode='A', size=256, len_shift=2,
                 use_source_injection=False,
                 use_dispersive=False,
                 use_dispersive_block=False):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.num_stages = num_stages
        self.share_weights = share_weights
        self.nC = dim
        self.size = size
        self.len_shift = len_shift
        self.use_source_injection = use_source_injection
        self.use_dispersive = use_dispersive

        # ParaEstimator：每个 stage 独立（即使 share_weights=True）
        self.rho_estimators = nn.ModuleList([
            ParaEstimator(in_nc=dim) for _ in range(num_stages)
        ])

        # Prior networks: WPO3D / KG
        prior_class = WaveMST_KG if use_kg else WaveMST_3D
        if share_weights:
            self.shared_prior = prior_class(
                dim=dim, stage=stage, num_blocks=num_blocks, mask_mode=mask_mode,
                use_dispersive_block=use_dispersive_block,
            )
            self.priors = None
        else:
            self.priors = nn.ModuleList([
                prior_class(
                    dim=dim, stage=stage, num_blocks=num_blocks, mask_mode=mask_mode,
                    use_dispersive_block=use_dispersive_block,
                )
                for _ in range(num_stages)
            ])
            self.shared_prior = None

        # 初始化卷积：融合 [shift_back(g), Phi] → [B, C, H, W]
        self.initial_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)

        # ── 模块 A：源项注入 ──
        if use_source_injection:
            self.source_convs = nn.ModuleList([
                nn.Conv2d(dim * 2, dim, 1, 1, 0)
                for _ in range(num_stages)
            ])

        # ── 模块 C：色散修正 ──
        if use_dispersive:
            from enhancement_ops import DispersionCorrector
            self.dispersion_corrs = nn.ModuleList([
                DispersionCorrector(dim)
                for _ in range(num_stages)
            ])

    def get_prior(self, k):
        """获取第 k 个 stage 的 prior network"""
        return self.shared_prior if self.share_weights else self.priors[k]

    def forward(self, g, input_mask):
        """
        Args:
            g:          [B, 1, H, W'] 测量值（W' = W + (C-1)*len_shift）
            input_mask: tuple (Phi, PhiPhiT)
                Phi:    [B, C, H, W] spatial mask
                PhiPhiT:[B, 1, H, W'] 预计算的 Phi*Phi^T

        Returns:
            list of [B, C, H, W]：每个 stage 的输出（length = num_stages）
        """
        Phi, PhiPhiT = input_mask

        # 预计算 shifted mask
        Phi_shift = shift_batch(Phi, self.len_shift)  # [B, C, H, W']

        # 初始化 f0：简单反投影
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)    # [B, C, H, W']
        f0 = shift_back_batch(temp_g, self.len_shift, self.size)  # [B, C, H, W]

        # 融合初始估计与 mask
        f = self.initial_conv(torch.cat([f0, Phi], dim=1))  # [B, C, H, W]

        # 预计算 Φ^T g（模块 A 用，只算一次）
        if self.use_source_injection:
            PhiT_g = mul_PhiT_residual(Phi_shift, g, self.len_shift, self.size)

        outputs = []
        for k in range(self.num_stages):
            # ── GD step ──
            rho_k = self.rho_estimators[k](f)          # [B, 1, 1, 1]
            Phi_f = mul_Phi_f(Phi_shift, f, self.len_shift)  # [B, 1, H, W']
            residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
            residual = residual.clamp(min=-10, max=10)  # 数值稳定
            z = f + rho_k * mul_PhiT_residual(
                Phi_shift, residual, self.len_shift, self.size
            )  # [B, C, H, W]

            # ── 模块 A：源项注入（prior 之前）──
            if self.use_source_injection:
                z = self.source_convs[k](torch.cat([z, PhiT_g], dim=1))

            # ── Prior step（WPO3D）──
            f = self.get_prior(k)(z, Phi)

            # ── 模块 C：色散修正（prior 之后）──
            if self.use_dispersive:
                f = self.dispersion_corrs[k](f)

            outputs.append(f)

        return outputs


class WaveMST_KG_Unfold(WaveMST_3D_Unfold):
    """Model 8 — KG 方程的 unfolding 版本，与 Model 7 唯一区别是 use_kg=True"""

    def __init__(self, dim=28, stage=2, num_blocks=None,
                 num_stages=5, share_weights=False,
                 mask_mode='A', size=256, len_shift=2,
                 use_source_injection=False,
                 use_dispersive=False,
                 use_dispersive_block=False):
        super().__init__(
            dim=dim, stage=stage, num_blocks=num_blocks,
            num_stages=num_stages, share_weights=share_weights,
            use_kg=True, mask_mode=mask_mode,
            size=size, len_shift=len_shift,
            use_source_injection=use_source_injection,
            use_dispersive=use_dispersive,
            use_dispersive_block=use_dispersive_block,
        )
