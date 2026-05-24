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

from model.mask_ops import MaskGateA, MaskKleinGordonD


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
                 use_swin=False, swin_window_size=64, swin_shift=False,
                 fbgw_mode='none'):
        super().__init__()
        self.dim = dim
        self.use_swin = use_swin
        self.swin_window_size = swin_window_size
        self.swin_shift = swin_shift
        self.fbgw_mode = fbgw_mode

        # 可学习物理参数
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.vs    = nn.Parameter(torch.tensor(1.0))
        self.vl    = nn.Parameter(torch.tensor(0.5))
        self.t     = nn.Parameter(torch.tensor(1.0))

        # 噪声-阻尼耦合系数
        self._lambda_sigma = nn.Parameter(torch.tensor(0.1))

        # mask 机制
        self.mask_mode = mask_mode
        if mask_mode == 'A':
            self.mask_op = MaskGateA(dim, eps=eps)
        elif mask_mode == 'D':
            self.mask_op = MaskKleinGordonD(dim, eps=eps)
        else:
            raise ValueError(f"mask_mode 必须是 'A' 或 'D'，得到 '{mask_mode}'")

        # FBGW 方案 B：可学习频带权重
        if fbgw_mode == 'learnable_band':
            self.num_bands_fbgw = 8
            self._band_weights = nn.Parameter(torch.ones(self.num_bands_fbgw))

        # 输出投影
        self.out_norm   = LayerNorm2d(dim)
        self.out_linear = nn.Conv2d(dim, dim, 1, bias=False)

    def _get_effective_params(self):
        alpha = F.softplus(self.alpha)
        vs    = F.softplus(self.vs)
        vl    = F.softplus(self.vl)
        t     = F.softplus(self.t)
        return alpha, vs, vl, t

    def _build_freq_grid(self, C, H, W, device):
        fc = torch.fft.fftfreq(C, device=device)[:, None, None]
        fh = torch.fft.fftfreq(H, device=device)[None, :, None]
        fw = torch.fft.rfftfreq(W, device=device)[None, None, :]
        return fc, fh, fw

    def _wave_modulate(self, u0_fft, v0_fft, alpha, vs, vl, t, C, H, W, device):
        """在频域做波动方程调制，返回 out_fft [B, C, H, W//2+1]。"""
        fc, fh, fw = self._build_freq_grid(C, H, W, device)
        pi2 = (2 * math.pi) ** 2
        omega_sq = pi2 * (vs ** 2 * (fh ** 2 + fw ** 2) + vl ** 2 * fc ** 2)

        eta = omega_sq - (alpha / 2) ** 2

        # 欠阻尼
        pos = eta.clamp(min=0)
        omega_d = torch.sqrt(pos + 1e-30)
        cos_term  = torch.cos(omega_d * t)
        sinc_term_pos = torch.sin(omega_d * t) / (omega_d + 1e-8)

        # 过阻尼
        neg = (-eta).clamp(min=0)
        gamma = torch.sqrt(neg + 1e-30)
        cosh_term = torch.cosh(gamma * t)
        sinch_term_neg = torch.sinh(gamma * t) / (gamma + 1e-8)

        is_under = (eta >= 0)
        cs   = torch.where(is_under, cos_term,  cosh_term)
        sinc = torch.where(is_under, sinc_term_pos, sinch_term_neg)

        decay = torch.exp(-alpha * t / 2)

        out_fft = decay * (u0_fft * cs + (v0_fft + alpha / 2 * u0_fft) * sinc)
        return out_fft, sinc, decay

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
        """全局 WPO（不分窗）"""
        B, C, H, W = x.shape
        alpha, vs, vl, t = self._get_effective_params()

        # 噪声感知阻尼
        if sigma is not None:
            lambda_sigma = F.softplus(self._lambda_sigma)
            alpha = alpha + lambda_sigma * sigma.mean()

        # mask 操作生成 u0, v0
        if self.mask_mode == 'A':
            u0, v0 = self.mask_op(x, mask_spatial)
            m_sq = None
        else:  # 'D'
            u0, v0, m_sq = self.mask_op(x, mask_spatial)

        # FFT 维度
        if FFT_PAD_TO_POW2:
            C_fft = _next_power_of_2(C)
            H_fft = H
            W_fft = W
        else:
            C_fft, H_fft, W_fft = C, H, W

        # 3D rFFT
        u0_fft = torch.fft.rfftn(u0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
        v0_fft = torch.fft.rfftn(v0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))

        # 频域调制
        out_fft, sinc_term, decay = self._wave_modulate(
            u0_fft, v0_fft, alpha, vs, vl, t, C_fft, H_fft, W_fft, x.device
        )

        # FBGW 频带引导加权
        out_fft = self._apply_fbgw(out_fft, u0_fft, sigma)

        # 3D irFFT
        out = torch.fft.irfftn(out_fft, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
        if C_fft != C:
            out = out[:, :C, :, :]

        # 方案 D：Born 修正
        if self.mask_mode == 'D' and m_sq is not None:
            out = self.mask_op.apply_correction(out, m_sq, sinc_term[:C], decay, C, H, W)

        # 输出投影
        out = self.out_norm(out)
        out = out * F.silu(x)
        out = self.out_linear(out)
        return out

    def _swin_forward(self, x, mask_spatial, sigma=None):
        """Swin 窗口 WPO：窗内独立传播 + shifted window"""
        B, C, H, W = x.shape
        ws = self.swin_window_size

        # 如果图像尺寸不大于窗大小，退化为全局 WPO
        if H <= ws and W <= ws:
            return self._global_forward(x, mask_spatial, sigma)

        # shift（奇数层偏移 ws//2）
        if self.swin_shift:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(2, 3))
            mask_spatial = torch.roll(mask_spatial, shifts=(-ws // 2, -ws // 2), dims=(2, 3))

        # 切窗：[B, C, H, W] -> [B*nH*nW, C, ws, ws]
        nH, nW = H // ws, W // ws
        x_win = x.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws, ws)
        m_win = mask_spatial.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws, ws)

        # 每个窗内做 WPO
        out_win = self._global_forward(x_win, m_win, sigma)

        # 重组：[B*nH*nW, C, ws, ws] -> [B, C, H, W]
        out = out_win.view(B, nH, nW, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        # 反 shift
        if self.swin_shift:
            out = torch.roll(out, shifts=(ws // 2, ws // 2), dims=(2, 3))

        return out

    def forward(self, x, mask_spatial, sigma=None):
        if self.use_swin:
            return self._swin_forward(x, mask_spatial, sigma)
        else:
            return self._global_forward(x, mask_spatial, sigma)


# ──────────────────────────────────────────────
# WPO3D Block = LN + WPO3D + Residual + LN + FFN + Residual
# ──────────────────────────────────────────────

class WPO3DBlock(nn.Module):
    def __init__(self, dim, mask_mode='A',
                 use_swin=False, swin_window_size=64, swin_shift=False,
                 fbgw_mode='none'):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo   = WPO3D(dim, mask_mode=mask_mode,
                           use_swin=use_swin, swin_window_size=swin_window_size,
                           swin_shift=swin_shift, fbgw_mode=fbgw_mode)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = FFN(dim)

    def forward(self, x, mask_spatial, sigma=None):
        x = x + self.wpo(self.norm1(x), mask_spatial, sigma=sigma)
        x = x + self.ffn(self.norm2(x))
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
                 use_swin_wpo=False, swin_window_size=64,
                 fbgw_mode='none'):
        super().__init__()
        self.dim   = dim
        self.stage = stage

        if use_kg:
            mask_mode = 'D'
        self.mask_mode = mask_mode

        # 输入嵌入
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([
                WPO3DBlock(dim_stage, mask_mode,
                           use_swin=use_swin_wpo,
                           swin_window_size=swin_window_size,
                           swin_shift=(j % 2 == 1),
                           fbgw_mode=fbgw_mode)
                for j in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            WPO3DBlock(dim_stage, mask_mode,
                       use_swin=use_swin_wpo,
                       swin_window_size=swin_window_size,
                       swin_shift=(j % 2 == 1),
                       fbgw_mode=fbgw_mode)
            for j in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion  = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)
            blocks  = nn.ModuleList([
                WPO3DBlock(dim_stage // 2, mask_mode,
                           use_swin=use_swin_wpo,
                           swin_window_size=swin_window_size,
                           swin_shift=(j % 2 == 1),
                           fbgw_mode=fbgw_mode)
                for j in range(num_blocks[stage - 1 - i])
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
                fea = blk(fea, mask_spatial, sigma=sigma)
            fea_encoder.append(fea)
            masks_enc.append(mask_spatial)
            fea = fea_down(fea)
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))

        # Bottleneck
        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial, sigma=sigma)

        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_enc[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial, sigma=sigma)

        return self.mapping(fea) + x


class WaveMST_KG(WaveMST_3D):
    """WaveMST_3D(use_kg=True) 的别名"""
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 mask_mode='A',
                 use_swin_wpo=False, swin_window_size=64,
                 fbgw_mode='none'):
        super().__init__(dim=dim, stage=stage, num_blocks=num_blocks,
                         mask_mode=mask_mode, use_kg=True,
                         use_swin_wpo=use_swin_wpo,
                         swin_window_size=swin_window_size,
                         fbgw_mode=fbgw_mode)
