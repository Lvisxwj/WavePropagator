"""
swap-AdaSpec.py — AdaSpec (Adaptive Spectral Filtering)
对应代码：version2/model/wpo3d.py::WPO3D._apply_fbgw
对应公式：(1.16) SNR 自适应 Wiener；(1.18) 学习版频带权重
颜色：背景 #f3f2f7（Part I）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaSpecSNR(nn.Module):
    """方案 A — 零参数 SNR 自适应频带加权."""

    def forward(self, out_fft, u0_fft, sigma=None):
        power = u0_fft.abs() ** 2
        sigma_sq = (sigma.mean() ** 2 if sigma is not None
                    else torch.tensor(0.01, device=out_fft.device))
        W = torch.sigmoid((power - sigma_sq) / (power + sigma_sq + 1e-6))  # (1.16)
        return out_fft * W


class AdaSpecBand(nn.Module):
    """方案 B — 可学习频带权重."""

    def __init__(self, num_bands=8):
        super().__init__()
        self.num_bands = num_bands
        self.band_weights = nn.Parameter(torch.ones(num_bands))

    def forward(self, out_fft):
        C, H, W_half = out_fft.shape[-3:]
        fc = torch.fft.fftfreq(C, device=out_fft.device)[:, None, None]
        fh = torch.fft.fftfreq(H, device=out_fft.device)[None, :, None]
        fw = torch.fft.rfftfreq(W_half * 2 - 1, device=out_fft.device)[None, None, :]
        mag = torch.sqrt(fc ** 2 + fh ** 2 + fw ** 2 + 1e-8)
        idx = (mag / (mag.max() + 1e-8) * (self.num_bands - 1)).long().clamp(0, self.num_bands - 1)
        W = F.softplus(self.band_weights)[idx]                           # (1.18)
        return out_fft * W
