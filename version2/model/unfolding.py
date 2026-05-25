"""
unfolding.py — A-HQS deep unfolding wrapper

每个 stage:
  1. 退化估计 -> delta_Phi, deg_weight, sigma
  2. Nesterov 动量外推
  3. 修正 GD step（Phi_eff = Phi + delta_Phi）
  4. 初始场净化（deg_weight 加权 + 残差）
  5. WPO 传播（sigma 控制阻尼）
  6. 局部精化（DWConv FFN）
  7. 三路残差输出（z + f_wave + f_local）
"""

import torch
import torch.nn as nn
from model.wpo3d import WaveMST_3D, WaveMST_KG#, LayerNorm2d
from model.degradation import DegradationEstimation, construct_degraded_mask
from model.refinement import LocalRefinement
from model.utils import (
    shift_batch, shift_back_batch,
    mul_Phi_f, mul_PhiT_residual,
    ParaEstimator,
)


class WPO_Unfold(nn.Module):
    """A-HQS Unfolding：退化估计 -> 动量 -> GD -> 净化 -> WPO -> 精化

    Args:
        dim, unet_stage, num_blocks: 传给 WaveMST_3D
        use_kg: True -> KG 方程
        num_stages: K（unfolding stage 数）
        share_weights: 共享 prior + degradation + refinement
        use_swin_wpo, swin_window_size: Swin-WPO 选项
        fbgw_mode: FBGW 选项
        size: crop_size（用于 shift_back）
    """

    def __init__(self, dim=28, unet_stage=3, num_blocks=None,
                 use_kg=False,
                 num_stages=5, share_weights=True,
                 use_swin_wpo=False, swin_window_size=64,
                 fbgw_mode='none',
                 size=256, len_shift=2,
                 use_ahqs=False,
                 debug=False,
                 debug_counter = 80):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.num_stages = num_stages
        self.share_weights = share_weights
        self.use_ahqs = use_ahqs

        self.debug = debug
        self.debug_counter = debug_counter
        self.forward_counter = 0

        self.nC = dim
        self.size = size
        self.len_shift = len_shift

        mask_mode = 'D' if use_kg else 'A'

        # ParaEstimator：每 stage 独立（即使 share_weights）
        self.rho_estimators = nn.ModuleList([
            ParaEstimator(in_nc=dim) for _ in range(num_stages)
        ])

        # Nesterov 动量系数：仅 A-HQS 模式使用
        if use_ahqs:
            self.betas = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(num_stages)
            ])

        # 退化估计
        if share_weights:
            self.deg_est = DegradationEstimation(dim)
        else:
            self.deg_ests = nn.ModuleList([
                DegradationEstimation(dim) for _ in range(num_stages)
            ])

        # Prior: WaveMST_3D / KG
        prior_class = WaveMST_KG if use_kg else WaveMST_3D
        prior_kwargs = dict(
            dim=dim, stage=unet_stage, num_blocks=num_blocks,
            mask_mode=mask_mode,
            use_swin_wpo=use_swin_wpo,
            swin_window_size=swin_window_size,
            fbgw_mode=fbgw_mode,
        )
        if share_weights:
            self.shared_prior = prior_class(**prior_kwargs)
            self.priors = None
        else:
            self.priors = nn.ModuleList([
                prior_class(**prior_kwargs) for _ in range(num_stages)
            ])
            self.shared_prior = None

        # 局部精化
        if share_weights:
            self.local_refine = LocalRefinement(dim)
        else:
            self.local_refines = nn.ModuleList([
                LocalRefinement(dim) for _ in range(num_stages)
            ])

        # z_clean 归一化（防止 deg_weight 累积放大）
        # self.z_norm = LayerNorm2d(dim)

        # 初始化卷积
        self.initial_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)

    def _get_deg_est(self, k):
        if self.share_weights:
            return self.deg_est
        else:
            return self.deg_ests[k]

    def _get_prior(self, k):
        if self.share_weights:
            return self.shared_prior
        else:
            return self.priors[k]

    def _get_refine(self, k):
        if self.share_weights:
            return self.local_refine
        else:
            return self.local_refines[k]

    def forward(self, g, input_mask):
        self.forward_counter += 1
        debug_flag = self.debug and (self.forward_counter % self.debug_counter == 0)
        """
        g:          [B, 1, H, W'] measurement
        input_mask: (Phi, PhiPhiT)
            Phi:    [B, C, H, W] spatial mask
            PhiPhiT:[B, 1, H, W'] 预计算

        Returns: list of [B, C, H, W]，每个 stage 的输出
        """
        Phi, PhiPhiT = input_mask
        Phi_shift = shift_batch(Phi, self.len_shift)

        # 预计算退化 mask（只算一次）
        Phi_star = construct_degraded_mask(Phi, self.len_shift)

        # 初始化
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)
        f0 = shift_back_batch(temp_g, self.len_shift, self.size)
        f = self.initial_conv(torch.cat([f0, Phi], dim=1))

        if self.use_ahqs:
            f_prev = f.clone()  # 动量用
        outputs = []

        for k in range(self.num_stages):
            # 1. 退化估计
            delta_Phi, deg_weight, sigma = self._get_deg_est(
                0 if self.share_weights else k
            )(f, Phi, Phi_star)

            # 2. GD step
            if self.use_ahqs:
                # A-HQS: Nesterov 动量 + Phi_eff 修正
                beta_k = torch.sigmoid(self.betas[k])
                f_input = f + beta_k * (f - f_prev)
                f_prev = f.detach().clone()
                Phi_eff_shift = shift_batch(Phi + delta_Phi, self.len_shift)
                rho_k = self.rho_estimators[k](f_input)
                Phi_f = mul_Phi_f(Phi_eff_shift, f_input, self.len_shift)
                residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
                residual = residual.clamp(-10, 10)
                z = f_input + rho_k * mul_PhiT_residual(
                    Phi_eff_shift, residual, self.len_shift, self.size
                )
            else:
                # GAP: 标准一阶梯度下降，使用原始 Phi
                rho_k = self.rho_estimators[k](f)
                Phi_f = mul_Phi_f(Phi_shift, f, self.len_shift)
                residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
                residual = residual.clamp(-10, 10)
                z = f + rho_k * mul_PhiT_residual(
                    Phi_shift, residual, self.len_shift, self.size
                )

            # 3. 退化加权 + LayerNorm（prior 输入预处理）
            z_clean = z * (1.0 + deg_weight)
            # z_clean = self.z_norm(z_clean)

            # 4. WPO 传播（内部全局残差 mapping(fea)+x 负责保留 z_clean）
            f = self._get_prior(k)(z_clean, Phi, sigma=sigma)

            # 5. 局部精化（残差加在 WPO 输出上）
            f = f + self._get_refine(k)(f)

            # Debug 打印
            if debug_flag and k == 0 and len(outputs) == 0:
                print(f"[DEBUG] z       : min={z.min():.4f} max={z.max():.4f} mean={z.mean():.4f}")
                print(f"[DEBUG] deg_w   : min={deg_weight.min():.4f} max={deg_weight.max():.4f}")
                print(f"[DEBUG] sigma   : {sigma.mean():.4f}")
                print(f"[DEBUG] z_clean : min={z_clean.min():.4f} max={z_clean.max():.4f}")
                print(f"[DEBUG] f       : min={f.min():.4f} max={f.max():.4f}")
                print(f"[DEBUG] rho_k   : {rho_k.mean():.4f}")
                if self.use_ahqs:
                    print(f"[DEBUG] beta_k  : {torch.sigmoid(self.betas[k]).item():.4f}")

            outputs.append(f)

        return outputs
