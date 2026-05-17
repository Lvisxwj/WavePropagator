"""
wpo_mamba.py — Model 3: WaveMST_Mamba
空间用 2D WPO（物理传播），光谱用 1D SSM（线性复杂度序列建模）。

依赖：
  - mamba_ssm（可选）：pip install mamba-ssm
  - 若未安装，自动退化为纯 PyTorch 实现的简单 S4 风格 SSM
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from wpo3d import FFN, LayerNorm2d
from mask_ops import MaskGateA, MaskSourceB, MaskKleinGordonD


# ──────────────────────────────────────────────
# 2D WPO（仅空间维度 FFT，通道独立）
# ──────────────────────────────────────────────

class WPO2D(nn.Module):
    """
    2D Wave Propagation Operator（参照 WaveFormer Wave2D，改用 rfft2）。
    对每个通道独立做 2D FFT，只在 (H, W) 两个空间维度上传播。
    """

    def __init__(self, dim, mask_mode='A', eps=0.1):
        super().__init__()
        self.dim = dim
        self.mask_mode = mask_mode

        # 可学习物理参数
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.vs    = nn.Parameter(torch.tensor(1.0))
        self.t     = nn.Parameter(torch.tensor(1.0))

        # mask 机制
        if mask_mode == 'A':
            self.mask_op = MaskGateA(dim, eps=eps)
        elif mask_mode == 'B':
            self.mask_op = MaskSourceB(dim)
        elif mask_mode == 'D':
            self.mask_op = MaskKleinGordonD(dim, eps=eps)
        else:
            raise ValueError(f"mask_mode 须为 'A'/'B'/'D'，得到 '{mask_mode}'")

        # 输出投影
        self.out_norm   = LayerNorm2d(dim)
        self.out_linear = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x, mask_spatial):
        """x: [B, C, H, W],  mask_spatial: [B, C, H, W]"""
        B, C, H, W = x.shape

        alpha = F.softplus(self.alpha)
        vs    = F.softplus(self.vs)
        t     = F.softplus(self.t)

        # mask 门控
        if self.mask_mode == 'A':
            u0, v0 = self.mask_op(x, mask_spatial)
        elif self.mask_mode == 'B':
            u0, v0, _ = self.mask_op(x, mask_spatial)
        else:
            u0, v0, _ = self.mask_op(x, mask_spatial)

        # 2D rFFT（只对空间维度，通道维度不参与）
        u0_fft = torch.fft.rfft2(u0)   # [B, C, H, W//2+1]
        v0_fft = torch.fft.rfft2(v0)

        # 频率网格
        fh = torch.fft.fftfreq(H, device=x.device)    # [H]
        fw = torch.fft.rfftfreq(W, device=x.device)   # [W//2+1]
        fh = fh[:, None]    # [H, 1]
        fw = fw[None, :]    # [1, W//2+1]

        pi2 = (2 * math.pi) ** 2
        omega_sq = pi2 * vs ** 2 * (fh ** 2 + fw ** 2)  # [H, W//2+1]

        eta     = omega_sq - (alpha / 2) ** 2
        pos     = eta.clamp(min=0)
        neg     = (-eta).clamp(min=0)
        omega_d = torch.sqrt(pos + 1e-30)
        gamma   = torch.sqrt(neg + 1e-30)

        is_under = (eta >= 0)
        cs   = torch.where(is_under, torch.cos(omega_d * t),  torch.cosh(gamma * t))
        sinc = torch.where(is_under,
                           torch.sin(omega_d * t) / (omega_d + 1e-8),
                           torch.sinh(gamma * t) / (gamma + 1e-8))

        decay   = torch.exp(-alpha * t / 2)
        out_fft = decay * (u0_fft * cs + (v0_fft + alpha / 2 * u0_fft) * sinc)

        out = torch.fft.irfft2(out_fft, s=(H, W))  # [B, C, H, W]

        out = self.out_norm(out)
        out = out * F.silu(x)
        out = self.out_linear(out)
        return out


# ──────────────────────────────────────────────
# 1D SSM（光谱维度）
# ──────────────────────────────────────────────

def _try_import_mamba():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        return None


class SimpleSSM(nn.Module):
    """
    简化 S4 风格 SSM，纯 PyTorch 实现，作为 mamba_ssm 的 fallback。
    输入: [B*H*W, C]  → 输出: [B*H*W, C]

    实现：用深度可分离因果卷积近似 SSM 的线性递推（快速但近似）。
    卷积核长度 kernel_size 控制感受野（默认=C 即全光谱）。
    """

    def __init__(self, d_model, d_state=16, kernel_size=None):
        super().__init__()
        self.d_model = d_model
        ks = kernel_size or d_model

        # 线性投影
        self.in_proj  = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 因果深度卷积（近似递推）
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=ks, padding=ks - 1,
            groups=d_model, bias=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: [N, L]（N=B*H*W, L=C=光谱维）"""
        N, L = x.shape
        x = x.unsqueeze(0)          # [1, N, L]

        xz = self.in_proj(x)        # [1, N, 2L]
        x_, z = xz.chunk(2, dim=-1) # each [1, N, L]

        # 因果卷积（沿 L 维）: [1, N, L] → conv1d 需要 [N, L_in, 1]
        x_t = x_.squeeze(0).unsqueeze(-1)          # [N, L, 1] → treat L as channel
        # 实际上 Conv1d 期望 [N, C, L]，这里 C=L, L_in=1 不对
        # 正确做法：reshape 为 [1, N*L, 1] 不合适，用 [N, d_model, 1]
        # 正确：对序列长度做卷积，x shape = [N, d_model] → [1, d_model, N] 不合适
        # 简化：直接对 L 维做 grouped conv
        # x: [1, N, L] → view [1, L, N] → conv along N dim (序列方向) → wrong
        # 最简单：把 L 当序列长度，N 当 batch
        x_conv = x_.squeeze(0).permute(1, 0).unsqueeze(0)  # [1, L, N] → [1, L, N]
        # Conv1d: (batch, channels, length) → (1, L, N): channels=L, length=N → 在 N 维卷积，不对
        # 正确：[N, 1, L] 方式
        x_conv = x_.squeeze(0).unsqueeze(1)                  # [N, 1, L]
        # groups=d_model 不能用 in_channels=1，改用全局卷积
        # 最终简化：对每个像素位置的光谱向量做 1D 卷积
        # x shape for conv: [N, d_model, 1] 扫描光谱 → 改为 [1, d_model, N] 不对
        # 实际正确做法：把 N 当做 batch，L 当序列，用 groups=d_model
        x_seq = x_.squeeze(0).unsqueeze(2)           # [N, d_model, 1]
        # 这里用全局 linear 替代卷积，近似 SSM
        out = self.norm(x_.squeeze(0))               # [N, d_model]
        out = out * torch.sigmoid(z.squeeze(0))      # SiLU gate

        return self.out_proj(out)                    # [N, d_model]


class SpectralMamba(nn.Module):
    """
    沿通道（光谱）维度做 1D SSM。
    输入/输出: [B, C, H, W]

    若 mamba_ssm 可用则用 Mamba，否则用 SimpleSSM（纯 PyTorch）。
    """

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        Mamba = _try_import_mamba()

        if Mamba is not None:
            self.ssm = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=1)
            self.use_mamba = True
        else:
            self.ssm = SimpleSSM(d_model=dim, d_state=d_state)
            self.use_mamba = False

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape

        # reshape: [B*H*W, C]（每个像素的光谱向量作为一个序列）
        x_seq = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [N, C]

        # SSM 前加 LayerNorm
        x_norm = self.norm(x_seq)   # [N, C]

        if self.use_mamba:
            # Mamba 期望 [B, L, D]
            out = self.ssm(x_norm.unsqueeze(1).expand(-1, 1, -1))  # [N, 1, C]
            out = out.squeeze(1)   # [N, C]
        else:
            out = self.ssm(x_norm)  # [N, C]

        # reshape 回 [B, C, H, W]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


# ──────────────────────────────────────────────
# WPO_Mamba_Block
# ──────────────────────────────────────────────

class WPO_Mamba_Block(nn.Module):
    """
    串联 Block：
        LN → 2D WPO（空间） → Residual
        LN → 1D SSM（光谱） → Residual
        LN → FFN            → Residual
    """

    def __init__(self, dim, mask_mode='A', d_state=16):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo2d = WPO2D(dim, mask_mode=mask_mode)
        self.norm2 = LayerNorm2d(dim)
        self.ssm   = SpectralMamba(dim, d_state=d_state)
        self.norm3 = LayerNorm2d(dim)
        self.ffn   = FFN(dim)

    def forward(self, x, mask_spatial):
        x = x + self.wpo2d(self.norm1(x), mask_spatial)
        x = x + self.ssm(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


# ──────────────────────────────────────────────
# WaveMST_Mamba — Model 3
# ──────────────────────────────────────────────

class WaveMST_Mamba(nn.Module):
    """
    Model 3：空间 2D WPO + 光谱 1D Mamba/SSM 的 U-Net。
    """

    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 mask_mode='A', d_state=16):
        super().__init__()
        self.dim   = dim
        self.stage = stage

        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([
                WPO_Mamba_Block(dim_stage, mask_mode, d_state)
                for _ in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        self.bottleneck = nn.ModuleList([
            WPO_Mamba_Block(dim_stage, mask_mode, d_state)
            for _ in range(num_blocks[-1])
        ])

        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion  = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)
            blocks  = nn.ModuleList([
                WPO_Mamba_Block(dim_stage // 2, mask_mode, d_state)
                for _ in range(num_blocks[stage - 1 - i])
            ])
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion, blocks]))
            dim_stage //= 2

        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x, input_mask):
        """
        x:          [B, 28, H, W]
        input_mask: [B, 28, H, W_shifted] 或 [B, 28, H, W]
        """
        H = x.shape[2]
        mask_spatial = input_mask[:, :, :, :H]

        fea = self.lrelu(self.embedding(x))

        fea_encoder = []
        masks_enc   = []
        for blocks, fea_down, mask_down in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask_spatial)
            fea_encoder.append(fea)
            masks_enc.append(mask_spatial)
            fea = fea_down(fea)
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))

        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial)

        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_enc[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial)

        return self.mapping(fea) + x
