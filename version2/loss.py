"""
loss.py — 损失函数与评估指标

函数：
    torch_psnr(img, ref)               — PSNR，逐通道平均  [C, H, W]
    torch_ssim(img, ref)               — SSIM              [C, H, W]
    torch_sam(img, ref)                — SAM（光谱角）      [C, H, W]
    rmse_loss(pred, gt)                — RMSE 训练损失      [B, C, H, W]
    count_params(model)                — 参数量 (M)
    count_flops(model, shape)          — FLOPs (G)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from torch.autograd import Variable


# ──────────────────────────────────────────────
# 训练损失
# ──────────────────────────────────────────────

def rmse_loss(pred, gt):
    """RMSE 损失（与 MST 一致）"""
    return torch.sqrt(F.mse_loss(pred, gt))


# ──────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────

def torch_psnr(img, ref):
    """
    PSNR，逐通道计算后平均。
    img, ref: [C, H, W]，值域 [0,1]
    """
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    total = 0.
    for i in range(nC):
        mse = torch.mean((img[i] - ref[i]) ** 2)
        total += 10 * torch.log10(torch.tensor(255. * 255.) / mse)
    return total / nC


# ──────────────────────────────────────────────
# SSIM
# ──────────────────────────────────────────────

def _gaussian(window_size, sigma):
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2d.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_compute(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def torch_ssim(img, ref, window_size=11):
    """
    SSIM。
    img, ref: [C, H, W]，值域 [0,1]
    """
    img4 = img.unsqueeze(0)
    ref4 = ref.unsqueeze(0)
    channel = img4.shape[1]
    window = _create_window(window_size, channel)
    if img4.is_cuda:
        window = window.cuda(img4.get_device())
    window = window.type_as(img4)
    return _ssim_compute(img4, ref4, window, window_size, channel)


# ──────────────────────────────────────────────
# SAM（Spectral Angle Mapper）
# ──────────────────────────────────────────────

def torch_sam(img, ref, eps=1e-8):
    """
    SAM（光谱角，弧度均值）。
    img, ref: [C, H, W]，值域 [0,1]
    返回标量 tensor（弧度）
    """
    img_t = img.permute(1, 2, 0)
    ref_t = ref.permute(1, 2, 0)
    dot   = torch.sum(img_t * ref_t, dim=-1)
    norm1 = torch.norm(img_t, dim=-1).clamp(min=eps)
    norm2 = torch.norm(ref_t, dim=-1).clamp(min=eps)
    cos_angle = (dot / (norm1 * norm2)).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos_angle).mean()


# ──────────────────────────────────────────────
# 参数量 / FLOPs
# ──────────────────────────────────────────────

def count_params(model):
    """返回可训练参数量（百万 M）"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6


def count_flops(model, input_shape=(1, 28, 256, 256), device='cuda'):
    """
    返回 FLOPs（十亿 G）。
    优先用 thop，若未安装则用 fvcore，均不可用则返回 None。
    """
    dummy = torch.randn(*input_shape).to(device)
    mask_dummy = torch.randn(1, 28, 256, 310).to(device)

    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy, mask_dummy), verbose=False)
        return flops / 1e9
    except ImportError:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (dummy, mask_dummy))
        return flops.total() / 1e9
    except ImportError:
        pass

    return None
