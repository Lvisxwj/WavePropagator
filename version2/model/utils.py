"""
utils.py — Deep Unfolding 框架工具函数

包含 CASSI 的 shift/shift_back（batch 版，向量化），Phi 乘法算子，
以及用于 GD step 的步长预测器 ParaEstimator。

参考：SSR/Utils.py（shift_3/shift_4）、SSR/Model.py（Para_Estimator）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Shift 操作（batch 版，向量化）
# ──────────────────────────────────────────────

def shift_batch(f, len_shift=2):
    """Batched shift: [B, C, H, W] -> [B, C, H, W + (C-1)*len_shift]

    每个波段 c 沿宽度方向右移 c * len_shift 像素。
    使用 scatter 向量化代替逐通道循环。
    """
    B, C, H, W = f.shape
    pad_w = (C - 1) * len_shift
    W_out = W + pad_w

    col_idx = torch.arange(W, device=f.device)
    offsets = torch.arange(C, device=f.device) * len_shift
    idx = col_idx.unsqueeze(0) + offsets.unsqueeze(1)  # [C, W]
    idx = idx.view(1, C, 1, W).expand(B, C, H, W)

    shifted = torch.zeros(B, C, H, W_out, device=f.device, dtype=f.dtype)
    shifted.scatter_(3, idx, f)
    return shifted


def shift_back_batch(f, len_shift=2, output_w=256):
    """Batched shift_back: [B, C, H, W'] -> [B, C, H, output_w]

    每个波段 c 从位置 c*len_shift 开始取 output_w 列。
    使用 gather 向量化代替逐通道循环。
    """
    B, C, H, W_in = f.shape

    col_idx = torch.arange(output_w, device=f.device)
    offsets = torch.arange(C, device=f.device) * len_shift
    idx = col_idx.unsqueeze(0) + offsets.unsqueeze(1)  # [C, output_w]
    idx = idx.view(1, C, 1, output_w).expand(B, C, H, output_w)

    return f.gather(3, idx)


# ──────────────────────────────────────────────
# Phi 乘法算子（CASSI 测量与反投影）
# ──────────────────────────────────────────────

def mul_Phi_f(Phi_shift, f, len_shift=2):
    """计算 Phi*f：shift(f) 与 Phi_shift 逐元素乘，再沿光谱维 sum。

    Phi_shift: [B, C, H, W']  已 shifted 的 mask
    f:         [B, C, H, W]   当前估计
    返回:      [B, 1, H, W']  测量值
    """
    f_shift = shift_batch(f, len_shift)
    Phi_f = Phi_shift * f_shift
    Phi_f = torch.sum(Phi_f, dim=1, keepdim=True)
    return Phi_f


def mul_PhiT_residual(Phi_shift, residual, len_shift=2, output_w=256):
    """计算 Phi^T r: 将 residual 广播到 C 通道，乘 Phi_shift，再 shift_back。

    Phi_shift: [B, C, H, W']
    residual:  [B, 1, H, W']
    返回:      [B, C, H, output_w]
    """
    nC = Phi_shift.shape[1]
    temp = residual.expand(-1, nC, -1, -1)
    PhiT = temp * Phi_shift
    PhiT = shift_back_batch(PhiT, len_shift, output_w)
    return PhiT


# ──────────────────────────────────────────────
# PhiPhiT 预计算
# ──────────────────────────────────────────────

def compute_PhiPhiT(mask3d, len_shift=2):
    """计算 Phi Phi^T 用于 GD step 的分母。

    mask3d:  [nC, H, W] 或 [B, nC, H, W]
    返回:    [1, H, W'] 或 [B, 1, H, W']
    """
    if mask3d.dim() == 3:
        mask3d_sq = mask3d.unsqueeze(0) ** 2
        shifted_sq = shift_batch(mask3d_sq, len_shift)
        PhiPhiT = torch.sum(shifted_sq, dim=1, keepdim=True)
        PhiPhiT = PhiPhiT.squeeze(0)
    else:
        mask3d_sq = mask3d ** 2
        shifted_sq = shift_batch(mask3d_sq, len_shift)
        PhiPhiT = torch.sum(shifted_sq, dim=1, keepdim=True)

    PhiPhiT[PhiPhiT == 0] = 1.0
    return PhiPhiT


# ──────────────────────────────────────────────
# 步长预测器
# ──────────────────────────────────────────────

class ParaEstimator(nn.Module):
    """从当前迭代值 f 预测 GD 步长 rho_k。

    输出 [B, 1, 1, 1] 标量。
    参考 SSR/Model.py Para_Estimator。
    """

    def __init__(self, in_nc=28, channel=32):
        super().__init__()
        self.fusion = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.bias = nn.Parameter(torch.FloatTensor([1.0]))
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1, padding=0, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fusion(x))
        x = self.avpool(x)
        x = self.mlp(x) + self.bias
        return x  # [B, 1, 1, 1]
