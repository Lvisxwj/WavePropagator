"""
wpo_smsa.py — Model 2: WaveMST_Parallel
WPO3D 并联 S-MSA，两路输出门控融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wpo3d import WPO3D, FFN, LayerNorm2d
from mst import MS_MSA


class WPO_SMSA_Block(nn.Module):
    """
    并联 Block：
        LN → [WPO3D || MS_MSA] → gate fusion → Residual
        LN → FFN → Residual

    WPO3D 用 mask_spatial [B,C,H,W]。
    MS_MSA 用 shift_mask  [B,C,H,W_shifted]。
    """

    def __init__(self, dim, mask_mode='A', fusion='gate', base_dim=28):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.wpo   = WPO3D(dim, mask_mode=mask_mode)
        heads = max(1, dim // base_dim)
        self.smsa  = MS_MSA(dim=dim, dim_head=base_dim, heads=heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = FFN(dim)
        self.fusion = fusion

        if fusion == 'gate':
            self.gate_net = nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1, bias=False),
                nn.Sigmoid(),
            )
        elif fusion == 'linear':
            self.fuse_linear = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, x, mask_spatial, shift_mask):
        """
        x:            [B, C, H, W]
        mask_spatial: [B, C, H, W]
        shift_mask:   [B, C, H, W_shifted]
        """
        residual = x
        x_n = self.norm1(x)

        # WPO 分支
        out_wpo = self.wpo(x_n, mask_spatial)  # [B, C, H, W]

        # S-MSA 分支（需要 channels-last 输入）
        x_last = x_n.permute(0, 2, 3, 1)   # [B, H, W, C]
        out_smsa = self.smsa(
            x_last,
            mask=shift_mask[:1]             # MS_MSA 只用 batch 中第0个 mask
        ).permute(0, 3, 1, 2)               # [B, C, H, W]

        # 融合
        if self.fusion == 'add':
            out = out_wpo + out_smsa
        elif self.fusion == 'gate':
            g = self.gate_net(torch.cat([out_wpo, out_smsa], dim=1))
            out = g * out_wpo + (1 - g) * out_smsa
        elif self.fusion == 'linear':
            out = self.fuse_linear(torch.cat([out_wpo, out_smsa], dim=1))
        else:
            out = out_wpo + out_smsa

        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class WaveMST_Parallel(nn.Module):
    """
    Model 2：WPO 并联 S-MSA 的 U-Net。

    forward 需要同时接收 shift_mask（给 MS_MSA）和 mask_spatial（给 WPO）。
    """

    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 mask_mode='A', fusion='gate'):
        super().__init__()
        self.dim      = dim
        self.stage    = stage
        self.base_dim = dim   # 记录基础 dim，用于 MS_MSA heads 计算

        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([
                WPO_SMSA_Block(dim_stage, mask_mode, fusion, base_dim=dim)
                for _ in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            smask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down, smask_down]))
            dim_stage *= 2

        self.bottleneck = nn.ModuleList([
            WPO_SMSA_Block(dim_stage, mask_mode, fusion, base_dim=dim)
            for _ in range(num_blocks[-1])
        ])

        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up  = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion_ = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)
            blocks  = nn.ModuleList([
                WPO_SMSA_Block(dim_stage // 2, mask_mode, fusion, base_dim=dim)
                for _ in range(num_blocks[stage - 1 - i])
            ])
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion_, blocks]))
            dim_stage //= 2

        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

    def forward(self, x, input_mask):
        """
        x:          [B, 28, H, W]
        input_mask: [B, 28, H, W_shifted]  shifted mask
        """
        H = x.shape[2]
        mask_spatial = input_mask[:, :, :, :H]   # [B, 28, H, W]
        shift_mask   = input_mask                 # [B, 28, H, W_shifted]

        fea = self.lrelu(self.embedding(x))

        fea_encoder  = []
        masks_spatial = []
        masks_shifted = []

        for blocks, fea_down, mask_down, smask_down in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask_spatial, shift_mask)
            fea_encoder.append(fea)
            masks_spatial.append(mask_spatial)
            masks_shifted.append(shift_mask)
            fea        = fea_down(fea)
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))
            shift_mask   = smask_down(shift_mask)

        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial, shift_mask)

        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_spatial[self.stage - 1 - i]
            shift_mask   = masks_shifted[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial, shift_mask)

        return self.mapping(fea) + x
