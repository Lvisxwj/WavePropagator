"""
swap.py — Part I 顶层: SWAP (Spectral WAve Propagator)
对应代码：version2/model/wpo3d.py::WaveMST_3D
对应公式：(1.9), (1.39) 完整 U-Net 骨架
颜色：背景 #f3f2f7（Part I）
"""

import torch
import torch.nn as nn


class SWAP(nn.Module):
    """
    U-Net backbone where each WPO3DBlock = LN + WPO3D + FFN
    Inputs:
        x            : [B, Λ, H, W]  净化后的语义场 z_clean
        mask_spatial : [B, Λ, H, W]  空间 mask Φ
        sigma        : [B, 1, 1, 1]  NLE 输出，控制 α_eff = α + λ_σ σ
    Output:
        f_wave       : [B, Λ, H, W]
    """

    def __init__(self, dim=28, stage=3, num_blocks=(2, 2, 2),
                 mask_mode='A', use_swin=False, swin_window_size=64,
                 fbgw_mode='snr_adaptive'):
        super().__init__()
        # ── Stem ───────────────────────────────────
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # ── Encoder ────────────────────────────────
        self.encoder = nn.ModuleList()
        d = dim
        for i in range(stage):
            blocks = nn.ModuleList(
                [SWAPBlock(d, mask_mode, use_swin, swin_window_size,
                           swin_shift=(j % 2 == 1), fbgw_mode=fbgw_mode)
                 for j in range(num_blocks[i])]
            )
            self.encoder.append(nn.ModuleList([
                blocks,
                nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),   # fea_down
                nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),   # mask_down
            ]))
            d *= 2

        # ── Bottleneck ─────────────────────────────
        self.bottleneck = nn.ModuleList([
            SWAPBlock(d, mask_mode, use_swin, swin_window_size,
                      swin_shift=(j % 2 == 1), fbgw_mode=fbgw_mode)
            for j in range(num_blocks[-1])
        ])

        # ── Decoder ────────────────────────────────
        self.decoder = nn.ModuleList()
        for i in range(stage):
            self.decoder.append(nn.ModuleList([
                nn.ConvTranspose2d(d, d // 2, 2, 2, 0),     # up
                nn.Conv2d(d, d // 2, 1, 1, bias=False),      # fusion
                nn.ModuleList(
                    [SWAPBlock(d // 2, mask_mode, use_swin, swin_window_size,
                               swin_shift=(j % 2 == 1), fbgw_mode=fbgw_mode)
                     for j in range(num_blocks[stage - 1 - i])]
                ),
            ]))
            d //= 2

        self.mapping = nn.Conv2d(dim, 28, 3, 1, 1, bias=False)

    def forward(self, x, mask_spatial, sigma=None):
        fea = self.act(self.embedding(x))
        enc, masks = [], []
        for blocks, fdown, mdown in self.encoder:
            for blk in blocks:
                fea = blk(fea, mask_spatial, sigma=sigma)
            enc.append(fea); masks.append(mask_spatial)
            fea = fdown(fea); mask_spatial = torch.sigmoid(mdown(mask_spatial))
        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial, sigma=sigma)
        for i, (up, fuse, blocks) in enumerate(self.decoder):
            fea = up(fea)
            fea = fuse(torch.cat([fea, enc[-1 - i]], dim=1))
            mask_spatial = masks[-1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial, sigma=sigma)
        return self.mapping(fea) + x   # global residual (1.39)


# 占位（详见 swap-Block.py / WPO3D 实际实现在 version2/model/wpo3d.py）
class SWAPBlock(nn.Module):  # noqa
    def __init__(self, *a, **k): super().__init__()
    def forward(self, x, mask, sigma=None): return x
