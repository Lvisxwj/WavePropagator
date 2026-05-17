"""
wpo3d_phys.py — Model 4: WaveMST_Phys (H2-α)

在 Model 0 (WaveMST_3D) 基础上：
  1. 把 3D rFFT 改为 2D rFFT（仅空间维）
  2. 把光谱方向可学习波速 vl·ωλ 替换为物理波数 k_phys(λ)=λ_min/λ_b
  3. 引入软硬先验混合比 γ，允许网络在物理值附近微调

mask_mode 支持 'A' / 'B' / 'D'，与 wpo3d.py 语义一致。
修正方案 B/D 使用 2D rFFT（与本模型的传播算子一致）。

数学基础：Helmholtz_HSI_Analysis.md §5（H2-α）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics import get_k_phys_for_dim
from mask_ops import MaskGateA, MaskSourceB, MaskKleinGordonD


# ──────────────────────────────────────────────
# LayerNorm（channels-first，复用 wpo3d.py 的设计）
# ──────────────────────────────────────────────

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ──────────────────────────────────────────────
# FFN（channels-first，与 wpo3d.py 一致）
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
# WPO3DPhys：核心算子
# ──────────────────────────────────────────────

class WPO3DPhys(nn.Module):
    """
    物理波数注入的 2D WPO（H2-α 核心）。

    色散关系：ω₀²(ω_xy, λ) = vs² · |ω_xy|² + k_eff²(λ)
    每个光谱通道是独立谐振子，固有频率由物理波数决定。

    mask_mode: 'A' / 'B' / 'D'
        A — 方案 A：初始振幅软门控（MaskGateA）
        B — 方案 B：Mask 作为源项（MaskSourceB），2D FFT 叠加
        D — 方案 D：Klein-Gordon Born 修正（MaskKleinGordonD），2D FFT 修正

    输入: x [B, C, H, W], mask_spatial [B, C, H, W]
    输出: [B, C, H, W]
    """

    def __init__(self, dim: int, mask_mode: str = 'A', mask_eps: float = 0.1):
        super().__init__()
        self.mask_mode = mask_mode

        # WPO 物理参数（softplus 保正）
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.vs    = nn.Parameter(torch.tensor(1.0))
        self.t     = nn.Parameter(torch.tensor(1.0))

        # 物理波数（固定 buffer）和可学习修正
        k_init = get_k_phys_for_dim(dim)
        self.register_buffer('k_phys', k_init)
        self.k_learn   = nn.Parameter(k_init.clone())
        self.gamma_raw = nn.Parameter(torch.tensor(-2.2))  # sigmoid→0.1

        # Mask 机制（替代原来内嵌的 phi/psi + 硬编码 mode A gate）
        if mask_mode == 'A':
            self.mask_op = MaskGateA(dim, eps=mask_eps)
        elif mask_mode == 'B':
            self.mask_op = MaskSourceB(dim)
        elif mask_mode == 'D':
            self.mask_op = MaskKleinGordonD(dim, eps=mask_eps)
        else:
            raise ValueError(f"mask_mode 必须是 'A', 'B', 'D'，得到 '{mask_mode}'")

        # 输出投影（与 WPO3D 一致）
        self.out_norm   = LayerNorm2d(dim)
        self.out_linear = nn.Conv2d(dim, dim, 1, bias=False)

    def _get_k_eff(self) -> torch.Tensor:
        gamma = torch.sigmoid(self.gamma_raw)
        return (1.0 - gamma) * self.k_phys + gamma * self.k_learn  # [C]

    def forward(self, x: torch.Tensor, mask_spatial: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 1. Mask 操作生成 u0, v0
        if self.mask_mode == 'A':
            u0, v0 = self.mask_op(x, mask_spatial)
            source = None
            m_sq   = None
        elif self.mask_mode == 'B':
            u0, v0, source = self.mask_op(x, mask_spatial)
            m_sq = None
        else:  # 'D'
            u0, v0, m_sq = self.mask_op(x, mask_spatial)
            source = None

        # 2. 2D rFFT（仅空间维）
        u0_fft = torch.fft.rfft2(u0)   # [B, C, H, W//2+1]
        v0_fft = torch.fft.rfft2(v0)

        # 3. 空间频率网格
        fh = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        fw = torch.fft.rfftfreq(W, device=device).view(1, 1, 1, W // 2 + 1)
        pi2 = (2.0 * math.pi) ** 2
        omega_xy_sq = pi2 * (fh ** 2 + fw ** 2)  # [1, 1, H, W//2+1]

        # 4. 物理波数
        k_eff = self._get_k_eff()               # [C]
        k_sq  = (k_eff ** 2).view(1, C, 1, 1)  # [1, C, 1, 1]

        # 5. WPO 参数
        alpha = F.softplus(self.alpha)
        vs    = F.softplus(self.vs)
        t     = F.softplus(self.t)

        # 6. 色散关系 ω₀² 和判别式 η
        omega0_sq = vs ** 2 * omega_xy_sq + k_sq  # [1, C, H, W//2+1]
        eta = omega0_sq - (alpha / 2.0) ** 2

        # 7. 欠/过阻尼分区
        is_under = eta >= 0.0
        sqrt_pos = torch.sqrt(eta.clamp(min=0.0) + 1e-30)
        sqrt_neg = torch.sqrt((-eta).clamp(min=0.0) + 1e-30)

        cs = torch.where(
            is_under,
            torch.cos(sqrt_pos * t),
            torch.cosh((sqrt_neg * t).clamp(max=20.0)),
        )
        sinc = torch.where(
            is_under,
            torch.sin(sqrt_pos * t) / (sqrt_pos + 1e-8),
            torch.sinh((sqrt_neg * t).clamp(max=20.0)) / (sqrt_neg + 1e-8),
        )

        # 8. 闭式解：decay · [u0·Cs + (v0 + α/2·u0)·Sn]
        decay   = torch.exp(-alpha * t / 2.0)
        out_fft = decay * (u0_fft * cs + (v0_fft + alpha / 2.0 * u0_fft) * sinc)

        # 9. 方案 B：叠加 2D 源项
        if self.mask_mode == 'B' and source is not None:
            src_fft = torch.fft.rfft2(source)
            out_fft = out_fft + src_fft * sinc * decay * self.mask_op.get_source_weight()

        # 10. iFFT
        out = torch.fft.irfft2(out_fft, s=(H, W))  # [B, C, H, W]

        # 11. 方案 D：Born 修正（2D FFT 版，与本模型的 2D 传播算子一致）
        if self.mask_mode == 'D' and m_sq is not None:
            source_d  = -m_sq * out
            src_d_fft = torch.fft.rfft2(source_d)
            corr_fft  = src_d_fft * sinc * decay
            correction = torch.fft.irfft2(corr_fft, s=(H, W))
            out = out + self.mask_op.kg_weight * correction

        # 12. 输出投影（LayerNorm → SiLU gate → Linear）
        out = self.out_norm(out)
        out = out * F.silu(x)
        out = self.out_linear(out)
        return out


# ──────────────────────────────────────────────
# WPO3DPhysBlock = LN + WPO3DPhys + Residual + LN + FFN + Residual
# ──────────────────────────────────────────────

class WPO3DPhysBlock(nn.Module):
    def __init__(self, dim: int, mask_mode: str = 'A'):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo   = WPO3DPhys(dim, mask_mode=mask_mode)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = FFN(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.wpo(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# WaveMST_Phys — Model 4（H2-α 完整模型）
# ──────────────────────────────────────────────

class WaveMST_Phys(nn.Module):
    """
    Model 4: H2-α — 物理波数注入 WPO。

    U-Net 结构与 Model 0 (WaveMST_3D) 完全相同，
    仅将内部 Block 替换为 WPO3DPhysBlock。

    mask_mode: 'A' / 'B' / 'D'，透传至所有 WPO3DPhysBlock。
    """

    def __init__(self, dim: int = 28, stage: int = 2,
                 num_blocks: list = None, mask_mode: str = 'A'):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.dim       = dim
        self.stage     = stage
        self.mask_mode = mask_mode

        # 输入嵌入
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks    = nn.ModuleList([
                WPO3DPhysBlock(dim_stage, mask_mode) for _ in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            WPO3DPhysBlock(dim_stage, mask_mode) for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion  = nn.Conv2d(dim_stage, dim_stage // 2, 1, bias=False)
            blocks  = nn.ModuleList([
                WPO3DPhysBlock(dim_stage // 2, mask_mode)
                for _ in range(num_blocks[stage - 1 - i])
            ])
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion, blocks]))
            dim_stage //= 2

        # 输出映射
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """
        x:          [B, 28, H, W]
        input_mask: [B, 28, H, W_shifted] 或 [B, 28, H, W]
        """
        H = x.shape[2]
        mask_spatial = input_mask[:, :, :, :H] if input_mask.shape[-1] > H else input_mask

        fea = self.lrelu(self.embedding(x))

        # Encoder
        fea_encoder = []
        masks_enc   = []
        for blocks, fea_down, mask_down in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask_spatial)
            fea_encoder.append(fea)
            masks_enc.append(mask_spatial)
            fea = fea_down(fea)
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))

        # Bottleneck
        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial)

        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_enc[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial)

        return self.mapping(fea) + x
