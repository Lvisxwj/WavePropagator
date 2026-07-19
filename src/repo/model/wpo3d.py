"""
wpo3d.py — WaveMST_3D 和 WaveMST_KG

核心：3D Wave Propagation Operator (WPO3D)
  - 各向异性阻尼波动方程的频域闭式解
  - 处理欠阻尼（cos/sin）和过阻尼（cosh/sinh）两种情况
  - Mask 软门控（方案 A，默认）或 Klein-Gordon Born 修正（方案 D）
  - FBGW 频带引导加权（可选）
  - Swin 窗口 WPO（可选）
  - 噪声感知阻尼（sigma 参数）

U-Net 骨架参照 MST，WPO3D 替代 S-MSA。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.mask_ops import MaskGateA, MaskKleinGordonD
from model.cmb import CMB, SimpleMix


# ──────────────────────────────────────────────
# LayerNorm（channels-first）
# ──────────────────────────────────────────────

class LayerNorm2d(nn.LayerNorm):
    """对 [B, C, H, W] 做 LayerNorm（在 C 维上）"""
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ──────────────────────────────────────────────
# FFN（参照 WaveFormer Mlp，channels-first）
# ──────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 核心：WPO3D
# ──────────────────────────────────────────────

def _next_power_of_2(n):
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# 全局开关：是否将 FFT 维度 pad 到 2 的幂
FFT_PAD_TO_POW2 = True


class WPO3D(nn.Module):
    """
    3D Wave Propagation Operator。

    输入: x [B, C, H, W],  mask_spatial [B, C, H, W]
    输出: [B, C, H, W]

    可学习参数（每层独立）：
        alpha — 阻尼系数
        vs    — 空间波速
        vl    — 光谱波速
        t     — 传播时间步长
        _lambda_sigma — 噪声-阻尼耦合系数

    参数用 softplus 保证正值。
    """

    def __init__(self, dim, mask_mode='A', eps=0.1,
                 fbgw_mode='none',
                 use_sicmb=True, use_perchannel=True,
                 use_spectral_wave=True,
                 use_mask_gate=True,
                 wpo_variant='full',
                 wave_param_mode='free', wave_basis_count=3):
        super().__init__()
        self.dim = dim
        self.fbgw_mode = fbgw_mode
        self.use_sicmb = use_sicmb
        self.use_perchannel = use_perchannel
        self.use_spectral_wave = bool(use_spectral_wave)
        self.use_mask_gate = bool(use_mask_gate)
        self.wpo_variant = wpo_variant
        self.wave_param_mode = str(wave_param_mode).lower()
        self.wave_basis_count = int(wave_basis_count)
        if wpo_variant not in ('full', 'legacy', 'dual'):
            raise ValueError(f"wpo_variant must be full/legacy/dual, got {wpo_variant!r}")
        if self.wave_param_mode not in ('free', 'symmetric_basis'):
            raise ValueError("wave_param_mode must be free/symmetric_basis")

        # full/dual 使用 WaveFormer 风格前端；legacy 直接传播输入场。
        if wpo_variant in ('full', 'dual'):
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
            self.proj = nn.Conv2d(dim, dim * 2, 1, bias=False)
        if wpo_variant == 'dual':
            self.dual_gate = nn.Conv2d(dim * 2, dim, 1, bias=True)
            nn.init.zeros_(self.dual_gate.weight)
            nn.init.zeros_(self.dual_gate.bias)

        # 可学习物理参数
        if self.wave_param_mode == 'symmetric_basis':
            if self.wave_basis_count < 2:
                raise ValueError("wave_basis_count must be >=2 for symmetric_basis")
            self.alpha_base = nn.Parameter(torch.tensor(0.1))
            self.vs_base = nn.Parameter(torch.tensor(1.0))
            self.alpha_coeff = nn.Parameter(torch.zeros(self.wave_basis_count))
            self.vs_coeff = nn.Parameter(torch.zeros(self.wave_basis_count))
        elif use_perchannel:
            self.alpha = nn.Parameter(torch.full((dim,), 0.1))  # [C]
            self.vs    = nn.Parameter(torch.full((dim,), 1.0))  # [C]
        else:
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.vs    = nn.Parameter(torch.tensor(1.0))
        if self.use_spectral_wave:
            self.vl = nn.Parameter(torch.tensor(0.5))
        else:
            # Exact ablation of the spectral second-derivative term.
            self.register_buffer('vl', torch.tensor(0.0))
        self.t     = nn.Parameter(torch.tensor(1.0))

        # 噪声-阻尼耦合系数
        self._lambda_sigma = nn.Parameter(torch.tensor(-2.0))

        # mask 机制
        self.mask_mode = mask_mode
        if mask_mode == 'A':
            self.mask_op = MaskGateA(dim, eps=eps)
        elif mask_mode == 'D':
            self.mask_op = MaskKleinGordonD(dim, eps=eps)
        else:
            raise ValueError(f"mask_mode 必须是 'A' 或 'D'，得到 '{mask_mode}'")

        # 输出门控
        if use_sicmb:
            self.sicmb = nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.Conv2d(dim, dim, 11, 1, 5, groups=dim, bias=False),
            )

        # FBGW 方案 B：可学习频带权重
        if fbgw_mode == 'learnable_band':
            self.num_bands_fbgw = 8
            self._band_weights = nn.Parameter(torch.ones(self.num_bands_fbgw))

        # 输出投影
        self.out_norm   = LayerNorm2d(dim)
        self.out_linear = nn.Conv2d(dim, dim, 1, bias=False)

    def _symmetric_basis(self, channels):
        freq = torch.fft.fftfreq(channels, device=self.alpha_base.device).abs()
        freq = freq / freq.max().clamp_min(1e-6)
        centers = torch.linspace(
            0.0, 1.0, self.wave_basis_count, device=freq.device, dtype=freq.dtype
        )
        width = 0.45 if self.wave_basis_count == 3 else 1.25 / self.wave_basis_count
        basis = torch.exp(-0.5 * ((freq[:, None] - centers[None, :]) / width) ** 2)
        return basis / basis.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _get_effective_params(self, channels=None):
        if self.wave_param_mode == 'symmetric_basis':
            channels = self.dim if channels is None else int(channels)
            basis = self._symmetric_basis(channels)
            alpha_raw = self.alpha_base + basis @ self.alpha_coeff
            vs_raw = self.vs_base + basis @ self.vs_coeff
            alpha = F.softplus(alpha_raw).view(-1, 1, 1)
            vs = F.softplus(vs_raw).view(-1, 1, 1)
        elif self.use_perchannel:
            alpha = F.softplus(self.alpha).view(-1, 1, 1)  # [C, 1, 1]
            vs    = F.softplus(self.vs).view(-1, 1, 1)     # [C, 1, 1]
            if channels is not None and alpha.shape[0] != int(channels):
                pad = int(channels) - alpha.shape[0]
                alpha = F.pad(alpha, (0, 0, 0, 0, 0, pad))
                vs = F.pad(vs, (0, 0, 0, 0, 0, pad))
        else:
            alpha = F.softplus(self.alpha)
            vs    = F.softplus(self.vs)
        vl = F.softplus(self.vl) if self.use_spectral_wave else self.vl
        t  = F.softplus(self.t)
        return alpha, vs, vl, t

    def _build_freq_grid(self, C, H, W, device):
        fc = torch.fft.fftfreq(C, device=device)[:, None, None]
        fh = torch.fft.fftfreq(H, device=device)[None, :, None]
        fw = torch.fft.rfftfreq(W, device=device)[None, None, :]
        return fc, fh, fw

    def _wave_modulate(self, u0_fft, v0_fft, alpha, vs, vl, t, C, H, W, device):
        """在频域做波动方程调制，返回 out_fft [B, C, H, W//2+1]。

        融合 decay 与 cosh/sinh 计算，避免 float32 中间溢出：
          decay*cosh(gamma*t) = [exp((gamma-alpha/2)*t) + exp(-(gamma+alpha/2)*t)] / 2
        两个指数项 ≤ 0，因此结果 ≤ 1，不溢出。

        torch.where 梯度安全：对非选中分支使用安全值（1.0），
        防止 sqrt(~0) 的梯度 1/(2*sqrt(~0)) → Inf 在 backward 传播 NaN。
        """
        fc, fh, fw = self._build_freq_grid(C, H, W, device)
        pi2 = (2 * math.pi) ** 2
        omega_sq = pi2 * (vs ** 2 * (fh ** 2 + fw ** 2) + vl ** 2 * fc ** 2)

        return self._solve_damped(u0_fft, v0_fft, omega_sq, alpha, t)

    @staticmethod
    def _solve_damped(u0_fft, v0_fft, omega_sq, alpha, t):
        """Stable closed-form damped-wave solve for a supplied dispersion surface."""

        eta = omega_sq - (alpha / 2) ** 2
        is_under = (eta >= 0)
        half_alpha_t = alpha * t / 2

        # 欠阻尼 (eta >= 0): cos/sin 有界，直接乘 decay 即可
        # 安全值：非欠阻尼区域用 1.0，避免 sqrt(~0) 梯度爆炸
        safe_pos = torch.where(is_under, eta, torch.ones_like(eta))
        omega_d = torch.sqrt(safe_pos.clamp(min=1e-12))
        exp_under = torch.exp(-half_alpha_t)
        cs_under   = exp_under * torch.cos(omega_d * t)
        sinc_under = exp_under * torch.sin(omega_d * t) / (omega_d + 1e-8)

        # 过阻尼 (eta < 0): 融合 decay 计算，避免 cosh/sinh 溢出
        # 安全值：非过阻尼区域用 1.0
        safe_neg = torch.where(is_under, torch.ones_like(eta), -eta)
        gamma = torch.sqrt(safe_neg.clamp(min=1e-12))
        # gamma ≤ alpha/2 恒成立，所以两个指数都 ≤ 0
        exp_p = torch.exp((gamma - alpha / 2) * t)    # ≤ 1
        exp_n = torch.exp(-(gamma + alpha / 2) * t)   # ≤ 1
        cs_over   = (exp_p + exp_n) / 2
        sinc_over = (exp_p - exp_n) / (2 * gamma + 1e-8)

        fused_cs   = torch.where(is_under, cs_under,  cs_over)
        fused_sinc = torch.where(is_under, sinc_under, sinc_over)

        # 输出已内含 decay
        out_fft = u0_fft * fused_cs + (v0_fft + alpha / 2 * u0_fft) * fused_sinc
        decay = torch.exp(-half_alpha_t)
        return out_fft, fused_sinc, decay

    def _pad_channel_params(self, alpha, vs, channels):
        if self.wave_param_mode == 'symmetric_basis':
            alpha, vs, _, _ = self._get_effective_params(channels)
            return alpha, vs
        if self.use_perchannel and alpha.shape[0] != channels:
            pad = channels - alpha.shape[0]
            return F.pad(alpha, (0, 0, 0, 0, 0, pad)), F.pad(vs, (0, 0, 0, 0, 0, pad))
        return alpha, vs

    def _dual_wave_forward(self, u0, v0, alpha, vs, vl, t):
        """Parallel 2D spatial wave and 1D spectral wave with learned fusion."""
        _, C, H, W = u0.shape
        pi2 = (2 * math.pi) ** 2

        # Spatial branch: independent 2D propagation for each wavelength.
        u_sp = torch.fft.rfft2(u0, s=(H, W), dim=(-2, -1))
        v_sp = torch.fft.rfft2(v0, s=(H, W), dim=(-2, -1))
        fh = torch.fft.fftfreq(H, device=u0.device)[None, :, None]
        fw = torch.fft.rfftfreq(W, device=u0.device)[None, None, :]
        omega_sp = pi2 * vs ** 2 * (fh ** 2 + fw ** 2)
        out_sp, _, _ = self._solve_damped(u_sp, v_sp, omega_sp, alpha, t)
        out_sp = torch.fft.irfft2(out_sp, s=(H, W), dim=(-2, -1))

        # Spectral branch: independent 1D propagation at every spatial position.
        C_fft = _next_power_of_2(C) if FFT_PAD_TO_POW2 else C
        alpha_fft, _ = self._pad_channel_params(alpha, vs, C_fft)
        u_spec = torch.fft.fft(u0, n=C_fft, dim=1)
        v_spec = torch.fft.fft(v0, n=C_fft, dim=1)
        fc = torch.fft.fftfreq(C_fft, device=u0.device)[:, None, None]
        omega_spec = pi2 * vl ** 2 * fc ** 2
        out_spec, _, _ = self._solve_damped(u_spec, v_spec, omega_spec, alpha_fft, t)
        out_spec = torch.fft.ifft(out_spec, n=C_fft, dim=1).real[:, :C]

        mix = torch.sigmoid(self.dual_gate(torch.cat([out_sp, out_spec], dim=1)))
        return mix * out_sp + (1.0 - mix) * out_spec

    def _apply_fbgw(self, out_fft, u0_fft, sigma):
        """频带引导加权，在 WPO 频域调制之后应用。"""
        if self.fbgw_mode == 'none':
            return out_fft

        if self.fbgw_mode == 'snr_adaptive':
            # 方案 A：基于信噪比（零参数）
            power = u0_fft.abs() ** 2
            sigma_sq = sigma.mean().item() ** 2 if sigma is not None else 0.01
            W = torch.sigmoid((power - sigma_sq) / (power + sigma_sq + 1e-6))
            return out_fft * W

        elif self.fbgw_mode == 'learnable_band':
            # 方案 B：可学习频带权重
            # 按 |omega| 分成 K 个频带
            C, H, W_half = out_fft.shape[-3], out_fft.shape[-2], out_fft.shape[-1]
            fc = torch.fft.fftfreq(C, device=out_fft.device)[:, None, None]
            fh = torch.fft.fftfreq(H, device=out_fft.device)[None, :, None]
            fw = torch.fft.rfftfreq(W_half * 2 - 1, device=out_fft.device)[None, None, :]
            freq_mag = torch.sqrt(fc ** 2 + fh ** 2 + fw ** 2 + 1e-8)
            # 归一化到 [0, K-1] 再量化
            freq_max = freq_mag.max()
            band_idx = (freq_mag / (freq_max + 1e-8) * (self.num_bands_fbgw - 1)).long()
            band_idx = band_idx.clamp(0, self.num_bands_fbgw - 1)
            weights = F.softplus(self._band_weights)
            W = weights[band_idx]  # [C, H, W_half]
            return out_fft * W

        return out_fft

    def _global_forward(self, x, mask_spatial, sigma=None):
        """全局 WPO（参照 WaveFormer Wave2D 结构）

        流程：dwconv → proj(dim→2dim) → split(x_wave, z_gate)
              x_wave → mask_op → FFT → wave_modulate → IFFT → norm
              output = norm(x_wave) * SiLU(z_gate) → out_linear
        """
        B, C, H, W = x.shape
        alpha, vs, vl, t = self._get_effective_params()

        # 噪声感知阻尼
        if sigma is not None:
            lambda_sigma = F.softplus(self._lambda_sigma)
            alpha = alpha + lambda_sigma * sigma.mean()

        if self.wpo_variant == 'legacy':
            x_wave, z_gate = x, x
        else:
            x_local = self.dwconv(x)
            x_proj = self.proj(x_local)
            x_wave, z_gate = x_proj.chunk(2, dim=1)

        # mask 操作生成 u0, v0（在 x_wave 上做，不是原始 x）
        if self.mask_mode == 'A':
            if self.use_mask_gate:
                u0, v0 = self.mask_op(x_wave, mask_spatial)
            else:
                # Ablation: remove MaskGateA's mask-conditioned amplitude gate
                # while keeping the learnable Phi/Psi initial-state maps.
                u0 = self.mask_op.phi(x_wave)
                v0 = self.mask_op.psi(x_wave)
            m_sq = None
        else:  # 'D'
            u0, v0, m_sq = self.mask_op(x_wave, mask_spatial)

        if self.wpo_variant == 'dual':
            if self.mask_mode != 'A':
                raise ValueError('dual WPO currently supports mask_mode=A only')
            out = self._dual_wave_forward(u0, v0, alpha, vs, vl, t)
            sinc_term = decay = None
        else:
            C_fft = _next_power_of_2(C) if FFT_PAD_TO_POW2 else C
            H_fft, W_fft = H, W
            u0_fft = torch.fft.rfftn(u0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
            v0_fft = torch.fft.rfftn(v0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
            alpha_fft, vs_fft = self._pad_channel_params(alpha, vs, C_fft)
            out_fft, sinc_term, decay = self._wave_modulate(
                u0_fft, v0_fft, alpha_fft, vs_fft, vl, t, C_fft, H_fft, W_fft, x.device
            )
            out_fft = self._apply_fbgw(out_fft, u0_fft, sigma)
            out = torch.fft.irfftn(out_fft, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
            if C_fft != C:
                out = out[:, :C]

        # 方案 D：Born 修正
        if self.mask_mode == 'D' and m_sq is not None:
            out = self.mask_op.apply_correction(out, m_sq, sinc_term[:C], decay, C, H, W)

        # 输出：norm(wave_out) * gate → linear
        out = self.out_norm(out)
        if self.use_sicmb:
            out = out * self.sicmb(z_gate)
        else:
            out = out * F.silu(z_gate)
        out = self.out_linear(out)
        return out

    def forward(self, x, mask_spatial, sigma=None):
        return self._global_forward(x, mask_spatial, sigma)


# ──────────────────────────────────────────────
# WPO3D Block = LN + WPO3D + Res + LN + CMB + Res [+ LN + SAB + Res]
# ──────────────────────────────────────────────

class WPO3DBlock(nn.Module):
    def __init__(self, dim, mask_mode='A',
                 fbgw_mode='none',
                 use_sicmb=True, use_perchannel=True,
                 use_spectral_wave=True,
                 use_mask_gate=True,
                 post_block='cmb', ffn_mult=2,
                 wpo_variant='full',
                 wave_param_mode='free', wave_basis_count=3):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo   = WPO3D(dim, mask_mode=mask_mode, fbgw_mode=fbgw_mode,
                           use_sicmb=use_sicmb, use_perchannel=use_perchannel,
                           use_spectral_wave=use_spectral_wave,
                           use_mask_gate=use_mask_gate,
                           wpo_variant=wpo_variant,
                           wave_param_mode=wave_param_mode,
                           wave_basis_count=wave_basis_count)
        self.post_block_type = post_block
        if post_block == 'cmb':
            self.norm2 = LayerNorm2d(dim)
            self.post  = CMB(dim)
        elif post_block == 'ffn':
            self.norm2 = LayerNorm2d(dim)
            self.post  = FFN(dim, mult=ffn_mult)
        elif post_block == 'simple':
            self.norm2 = LayerNorm2d(dim)
            self.post  = SimpleMix(dim)
        elif post_block == 'none':
            pass
        else:
            raise ValueError(f"post_block 须为 cmb/ffn/simple/none，得到 '{post_block}'")

    def forward(self, x, mask_spatial, sigma=None):
        x = x + self.wpo(self.norm1(x), mask_spatial, sigma=sigma)
        if self.post_block_type != 'none':
            x = x + self.post(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# WaveMST_3D — 主推模型
# ──────────────────────────────────────────────

class WaveMST_3D(nn.Module):
    """
    U-Net 骨架（参照 MST），WPO3D Block 替代 S-MSA。

    新增：FBGW、Swin-WPO、sigma 参数接口。
    """

    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 mask_mode='A', use_kg=False,
                 fbgw_mode='none',
                 use_sicmb=True, use_perchannel=True,
                 use_spectral_wave=True,
                 use_mask_gate=True,
                 post_block='cmb', ffn_mult=2,
                 wpo_variant='full', gradient_checkpointing=False,
                 wave_param_mode='free', wave_basis_count=3):
        super().__init__()
        self.dim   = dim
        self.stage = stage
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if use_kg:
            mask_mode = 'D'
        self.mask_mode = mask_mode

        # 输入嵌入
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        def _make_block(d):
            return WPO3DBlock(d, mask_mode,
                              fbgw_mode=fbgw_mode,
                              use_sicmb=use_sicmb,
                              use_perchannel=use_perchannel,
                              use_spectral_wave=use_spectral_wave,
                              use_mask_gate=use_mask_gate,
                              post_block=post_block,
                              ffn_mult=ffn_mult,
                              wpo_variant=wpo_variant,
                              wave_param_mode=wave_param_mode,
                              wave_basis_count=wave_basis_count)

        def _run_block(block, features, spatial_mask, noise):
            if self.gradient_checkpointing and self.training and noise is None:
                return checkpoint(block, features, spatial_mask)
            return block(features, spatial_mask, sigma=noise)

        self._run_block = _run_block

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([_make_block(dim_stage) for _ in range(num_blocks[i])])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([_make_block(dim_stage) for _ in range(num_blocks[-1])])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion  = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)
            blocks  = nn.ModuleList([
                _make_block(dim_stage // 2) for _ in range(num_blocks[stage - 1 - i])
            ])
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion, blocks]))
            dim_stage //= 2

        # 输出映射
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x, input_mask, sigma=None):
        """
        x:          [B, 28, H, W]
        input_mask: [B, 28, H, W_shifted] 或 [B, 28, H, W] spatial mask
        sigma:      [B, 1, 1, 1] 噪声水平（可选）
        """
        H = x.shape[2]
        if input_mask.shape[-1] > H:
            mask_spatial = input_mask[:, :, :, :H]
        else:
            mask_spatial = input_mask

        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks_enc   = []
        for blocks, fea_down, mask_down in self.encoder_layers:
            for blk in blocks:
                fea = self._run_block(blk, fea, mask_spatial, sigma)
            fea_encoder.append(fea)
            masks_enc.append(mask_spatial)
            fea = fea_down(fea)
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))

        # Bottleneck
        for blk in self.bottleneck:
            fea = self._run_block(blk, fea, mask_spatial, sigma)

        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_enc[self.stage - 1 - i]
            for blk in blocks:
                fea = self._run_block(blk, fea, mask_spatial, sigma)

        return self.mapping(fea) + x


class WaveMST_KG(WaveMST_3D):
    """WaveMST_3D(use_kg=True) 的别名"""
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 mask_mode='A',
                 fbgw_mode='none',
                 use_sicmb=True, use_perchannel=True,
                 use_spectral_wave=True,
                 post_block='cmb', ffn_mult=2,
                 wpo_variant='full', gradient_checkpointing=False):
        super().__init__(dim=dim, stage=stage, num_blocks=num_blocks,
                         mask_mode=mask_mode, use_kg=True,
                         fbgw_mode=fbgw_mode,
                         use_sicmb=use_sicmb, use_perchannel=use_perchannel,
                         use_spectral_wave=use_spectral_wave,
                         post_block=post_block, ffn_mult=ffn_mult,
                         wpo_variant=wpo_variant,
                         gradient_checkpointing=gradient_checkpointing)


