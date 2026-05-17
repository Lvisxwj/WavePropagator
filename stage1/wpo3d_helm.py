"""
wpo3d_helm.py — Model 6: WaveMST_Helm (H2-γ，主推方案)

三合一统一框架：
  Step 1. Mask 初始软门控    — CASSI 激励编码（继承 H2-α）
  Step 2. 物理波数 WPO 传播  — 时域动态传播，k(λ) 决定振荡频率
  Step 3. Beer-Lambert 吸收  — 输出幅值修正，mask 控制透射率

完整前向公式（Helmholtz_HSI_Analysis.md §6 公式 6.7）：
  f_out = exp(-κ₀·(1-M)·2π·L/λ) · IFFT[decay·(û0M·Cs + (v̂0M + α/2·û0M)·Sn)]

Model 6 = Model 4 (WaveMST_Phys) + BeerLambertAbsorption（每 Block 末尾追加）

mask_mode: 'A' / 'B' / 'D'，透传至 WPO3DPhysBlock（Step 1/2）。
Step 3 (BeerLambertAbsorption) 始终使用 mask，不受 mask_mode 影响。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 Model 4 的所有模块
from wpo3d_phys import WPO3DPhysBlock, LayerNorm2d, FFN
from physics import get_inv_lambda_for_dim
from helmholtz_ops import BeerLambertAbsorption


# ──────────────────────────────────────────────
# WPO3DHelmBlock：在 WPO3DPhysBlock 后追加 Beer-Lambert
# ──────────────────────────────────────────────

class WPO3DHelmBlock(nn.Module):
    """
    H2-γ Block：
      1. WPO3DPhysBlock（含 Step 1 Mask 门控 + Step 2 物理波数 WPO + FFN）
      2. BeerLambertAbsorption（Step 3 波长依赖吸收）

    mask_mode 透传给 WPO3DPhysBlock 控制 Step 1 的 mask 应用方式。
    """

    def __init__(self, dim: int, mask_mode: str = 'A'):
        super().__init__()
        # Step 1+2：物理波数 WPO（mask_mode 软编码）
        self.wpo_block = WPO3DPhysBlock(dim, mask_mode=mask_mode)

        # Step 3：Beer-Lambert 吸收（始终使用 mask，不受 mask_mode 影响）
        inv_lam = get_inv_lambda_for_dim(dim)
        self.absorption = BeerLambertAbsorption(dim, inv_lambda_init=inv_lam,
                                                init_kappa=0.5, init_L=1.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.wpo_block(x, mask)
        x = self.absorption(x, mask)
        return x


# ──────────────────────────────────────────────
# WaveMST_Helm — Model 6（H2-γ 主推方案）
# ──────────────────────────────────────────────

class WaveMST_Helm(nn.Module):
    """
    Model 6: H2-γ — 三合一主推方案。

    U-Net 结构与 Model 0/4/5 完全相同，
    每个 Block 替换为 WPO3DHelmBlock（WPO + Beer-Lambert）。

    mask_mode: 'A' / 'B' / 'D'，透传至所有 WPO3DHelmBlock。
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
                WPO3DHelmBlock(dim_stage, mask_mode) for _ in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            WPO3DHelmBlock(dim_stage, mask_mode) for _ in range(num_blocks[-1])
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion  = nn.Conv2d(dim_stage, dim_stage // 2, 1, bias=False)
            blocks  = nn.ModuleList([
                WPO3DHelmBlock(dim_stage // 2, mask_mode)
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
