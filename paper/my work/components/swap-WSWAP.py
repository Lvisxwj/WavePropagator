"""
swap-WSWAP.py — W-SWAP (Windowed SWAP，可选)
对应代码：version2/model/wpo3d.py::WPO3D._swin_forward
对应公式：(1.26), (1.27) 窗内独立波传播 + shifted window
颜色：背景 #f3f2f7（Part I）
"""

import torch
import torch.nn as nn


class WSWAP(nn.Module):
    """
    Inputs:
        x            : [B, Λ, H, W]
        mask_spatial : [B, Λ, H, W]
        sigma        : [B, 1, 1, 1]
    Outputs:
        f_wave       : [B, Λ, H, W]
    """

    def __init__(self, swap_global_forward, window_size=64, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.swap_global = swap_global_forward   # 引用 SWAP 全局前向（公式 (1.9)）

    def forward(self, x, mask_spatial, sigma=None):
        B, C, H, W = x.shape
        ws = self.window_size
        if H <= ws and W <= ws:
            return self.swap_global(x, mask_spatial, sigma)

        if self.shift:
            x = torch.roll(x, (-ws // 2, -ws // 2), dims=(2, 3))
            mask_spatial = torch.roll(mask_spatial, (-ws // 2, -ws // 2), dims=(2, 3))

        nH, nW = H // ws, W // ws
        x_win = x.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws, ws)
        m_win = mask_spatial.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws, ws)
        out_win = self.swap_global(x_win, m_win, sigma)
        out = out_win.view(B, nH, nW, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if self.shift:
            out = torch.roll(out, (ws // 2, ws // 2), dims=(2, 3))
        return out
