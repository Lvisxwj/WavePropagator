"""
helm_pure.py — Model 5: Helmholtzformer (H1-γ)

纯稳态亥姆霍兹方程，无时间演化：
    f = IFFT[ FFT(M·s) / (k²(λ) - |ω|² + iε) ]

每一层做"源场编码 → mask 调制 → 亥姆霍兹逆算子"循环。
作为 Model 6 (H2-γ) 的消融基准：稳态共振 vs 动态传播。

mask_mode: 'A' / 'B'
    A — 软门控源场：s_gated = s * (eps + (1-eps)*mask)，亥姆霍兹算子不再内部乘 mask
    B — 源项调制（当前默认）：亥姆霍兹算子内部执行 ms = mask * s
    注意：方案 D（Klein-Gordon Born 修正）依赖波动方程中间量，
          与亥姆霍兹静态算子不兼容，不支持，传入 'D' 会抛出 ValueError。

数学基础：Helmholtz_HSI_Analysis.md §3（H1-γ）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics import get_k_phys_for_dim
from helmholtz_ops import HelmholtzInverseOp


# ──────────────────────────────────────────────
# LayerNorm / FFN（与 wpo3d_phys.py 一致）
# ──────────────────────────────────────────────

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


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
# HelmBlock = LN + (源场编码 + HelmholtzInverse + SiLU gate) + Residual + LN + FFN + Residual
# ──────────────────────────────────────────────

class HelmBlock(nn.Module):
    """
    亥姆霍兹 Block。

    前向流程（mask_mode='B'，原始行为）：
      1. LN(x)
      2. s = source_encoder(x_norm)
      3. z = gate_proj(x_norm)
      4. f = HelmholtzInverseOp(s, mask)   — 内部执行 mask * s
      5. out = f * silu(z) → out_proj
      6. x = x + out
      7. x = x + FFN(LN(x))

    mask_mode='A'：
      4. s_gated = s * (0.1 + 0.9 * mask)   — 源场软门控
         f = HelmholtzInverseOp(s_gated, ones)  — 算子内部不再重复乘 mask
    """

    def __init__(self, dim: int, mask_mode: str = 'A'):
        super().__init__()
        if mask_mode == 'D':
            raise ValueError(
                "Helmholtzformer 不支持 mask_mode='D'（Klein-Gordon Born 修正依赖波动方程"
                "中间量，与静态亥姆霍兹算子不兼容）。请使用 'A' 或 'B'。"
            )
        self.mask_mode = mask_mode
        self.mask_gate_eps = 0.1   # mode A 软门控 epsilon

        self.norm1 = LayerNorm2d(dim)

        # 源场编码器（DWConv + GELU + PW）
        self.source_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

        # 门控分支
        self.gate_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj  = nn.Conv2d(dim, dim, 1, bias=False)

        # 亥姆霍兹逆算子
        k_init = get_k_phys_for_dim(dim)
        self.helm_op = HelmholtzInverseOp(dim, k_init=k_init)

        self.norm2 = LayerNorm2d(dim)
        self.ffn   = FFN(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        s = self.source_encoder(x_norm)
        z = self.gate_proj(x_norm)

        if self.mask_mode == 'A':
            # 源场软门控，传入全 1 让算子内部跳过重复乘 mask
            gate    = self.mask_gate_eps + (1.0 - self.mask_gate_eps) * mask
            s_gated = s * gate
            f = self.helm_op(s_gated, torch.ones_like(mask))
        else:  # 'B'
            # 算子内部执行 mask * s（原始行为）
            f = self.helm_op(s, mask)

        out = self.out_proj(f * F.silu(z))
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# Helmholtzformer — Model 5（H1-γ 完整模型）
# ──────────────────────────────────────────────

class Helmholtzformer(nn.Module):
    """
    Model 5: H1-γ — 纯稳态亥姆霍兹模型。

    U-Net 结构与 Model 0/4 完全相同，每个 Block 用 HelmBlock。
    用于消融实验：对比"稳态共振"vs"动态 WPO 传播"的贡献差。

    mask_mode: 'A' / 'B'（不支持 'D'，详见 HelmBlock）
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
                HelmBlock(dim_stage, mask_mode) for _ in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            HelmBlock(dim_stage, mask_mode) for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion  = nn.Conv2d(dim_stage, dim_stage // 2, 1, bias=False)
            blocks  = nn.ModuleList([
                HelmBlock(dim_stage // 2, mask_mode)
                for _ in range(num_blocks[stage - 1 - i])
            ])
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion, blocks]))
            dim_stage //= 2

        # 输出映射
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
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
