"""
ml_layers.py — 三种 ML 层 + 统一接口

选项 1: ML_DWConvCA            — DWConv + Channel Attention（轻量基线）
选项 2: ML_WSSA                — Window-based Spectral Self-Attention（参照 SSR）
选项 3: ML_FreqBandAttention   — 小波分频 + 频带内光谱 Attention（自主设计）

统一接口: build_ml_layer(ml_type, dim) → nn.Module
  forward(x) → x，输入输出 shape 都是 [B, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 选项 1：DWConv + Channel Attention
# ──────────────────────────────────────────────

class ML_DWConvCA(nn.Module):
    """DWConv + SE-style Channel Attention

    DWConv3×3 提取空间局部特征 → GELU → Conv1×1 通道混合
    → SE-style Channel Attention（squeeze-excitation）

    参数量：~3×dim² + 2×dim²/r ≈ 3.5×dim²
    dim=28 时约 2.7K 参数
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # DWConv
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),                     # 通道混合
        )
        # SE channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.spatial(x)
        attn = self.se(feat)
        return feat * attn


# ──────────────────────────────────────────────
# 选项 2：Window-based Spectral Self-Attention
# ──────────────────────────────────────────────

class ML_WSSA(nn.Module):
    """Window-based Spectral Self-Attention

    参照 SSR (CVPR 2024) 的 WSSA 设计。
    空间切成 M×M 窗，每个窗内做完整光谱 attention。

    SSR 原版细节：
    - Q, K, V 由 Conv2d 投影
    - Q 做 in-place scaling (dim_head ** -0.5)
    - attention 在 window_pixels 维度计算（空间位置间的注意力）
    - 可选 shift（cyclic roll）做跨窗信息交换

    我们的实现保留核心设计，去掉 einops 依赖，增加 shift 支持。

    参数量：~6×dim² ≈ 4.7K（dim=28）
    """

    def __init__(self, dim, window_size=8, shift=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift
        self.shift_size = window_size // 2

        # Q, K, V 投影
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.scale = dim ** -0.5

        # 初始化
        for m in [self.qkv, self.proj]:
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        M = self.window_size

        # cyclic shift（SSR 风格跨窗信息交换）
        if self.shift:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            x_shifted = x

        # Q, K, V
        qkv = self.qkv(x_shifted)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # 各 [B, C, H, W]

        # 空间切窗：[B, C, H, W] → [B*nH*nW, C, M, M]
        nH, nW = H // M, W // M

        def window_partition(t):
            # [B, C, H, W] → [B, C, nH, M, nW, M] → [B*nH*nW, C, M, M]
            return (t.view(B, C, nH, M, nW, M)
                     .permute(0, 2, 4, 1, 3, 5)
                     .reshape(B * nH * nW, C, M, M))

        q_w = window_partition(q)  # [BnW, C, M, M]
        k_w = window_partition(k)
        v_w = window_partition(v)

        # 展平空间：[BnW, C, M²]
        BnW = q_w.shape[0]
        q_flat = q_w.reshape(BnW, C, M * M)
        k_flat = k_w.reshape(BnW, C, M * M)
        v_flat = v_w.reshape(BnW, C, M * M)

        # SSR 风格：attention 在空间位置间（M² × M²），C 是 token 维度
        # sim[i,j] = sum_c q[c,i] * k[c,j]
        # 即 [BnW, M², M²] = [BnW, M², C] @ [BnW, C, M²]
        q_t = q_flat.permute(0, 2, 1)  # [BnW, M², C]
        k_t = k_flat.permute(0, 2, 1)  # [BnW, M², C]
        v_t = v_flat.permute(0, 2, 1)  # [BnW, M², C]

        q_t = q_t * self.scale
        attn = torch.bmm(q_t, k_t.transpose(1, 2))  # [BnW, M², M²]
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v_t)  # [BnW, M², C]

        # 重组：[BnW, M², C] → [BnW, C, M, M] → [B, C, H, W]
        out = out.permute(0, 2, 1).reshape(BnW, C, M, M)
        out = (out.view(B, nH, nW, C, M, M)
                  .permute(0, 3, 1, 4, 2, 5)
                  .reshape(B, C, H, W))

        # reverse cyclic shift
        if self.shift:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return self.proj(out)


# ──────────────────────────────────────────────
# 选项 3：Frequency-Band Attention (FBA)
# ──────────────────────────────────────────────

class ML_FreqBandAttention(nn.Module):
    """小波分频 + 频带内光谱 Self-Attention

    设计原理：
    1. 2D Haar DWT 把空间特征分解为 4 个频率子带 (LL, LH, HL, HH)
    2. 每个子带内独立做全光谱 Self-Attention（C×C attention）
    3. 不同子带有可学习的权重（低频全局、高频局部）
    4. 子带间交互（分组 Conv1×1）+ 逆小波重建

    物理动机：
    - WPO 的不同频率分量有不同传播特性（低频振荡传播、高频阻尼衰减）
    - FBA 按频带分别学习光谱相关性，与 WPO 的频域调制天然互补
    - WPO 管"频率内空间传播"，FBA 管"频率内光谱交互"

    文献支撑：
    - SSR (CVPR 2024)：空间窗内光谱 attention
    - Specformer (ECCV 2024)：频域注意力
    - HFMNet (IJCAI 2024)：双维度频域调制
    - 本方案：小波分频 + 频带内光谱 attention（新组合）

    复杂度：O(HW·C²/4 × 4) = O(HW·C²)，但子带空间减半，实际比 WSSA 轻约 60%
    参数量：~8×dim²（QKV + proj + band_interact）
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 共享 QKV 投影（4 子带共享，节省参数）
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        # 子带自适应权重：LL 最大，HH 最小
        self.band_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.5, 0.25]))

        # 子带间交互（分组 Conv1×1，4 组各处理 dim 通道）
        self.band_interact = nn.Conv2d(dim * 4, dim * 4, 1, groups=4, bias=False)

        self.scale = dim ** -0.5

        # 初始化
        for m in [self.qkv, self.proj]:
            nn.init.trunc_normal_(m.weight, std=0.02)

    @staticmethod
    def dwt2d(x):
        """2D Haar 小波变换（无参数，完美重建）

        输入: [B, C, H, W]  (H, W 必须是偶数)
        输出: (LL, LH, HL, HH)，各 [B, C, H/2, W/2]
        """
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 + x01 - x10 - x11) * 0.5
        HL = (x00 - x01 + x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5
        return LL, LH, HL, HH

    @staticmethod
    def idwt2d(LL, LH, HL, HH):
        """2D Haar 逆小波变换（完美重建）

        输入: 各 [B, C, H/2, W/2]
        输出: [B, C, H, W]
        """
        B, C, H2, W2 = LL.shape
        out = torch.zeros(B, C, H2 * 2, W2 * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
        out[:, :, 0::2, 1::2] = (LL + LH - HL - HH) * 0.5
        out[:, :, 1::2, 0::2] = (LL - LH + HL - HH) * 0.5
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5
        return out

    def spectral_attention(self, x, weight):
        """单个频率子带内的全光谱 Self-Attention

        x: [B, C, H', W']（子带，空间尺寸减半）
        weight: 标量，控制该子带 attention 强度

        每个空间位置的 C 维光谱向量作为 token，计算 C×C attention。
        """
        B, C, H, W = x.shape

        qkv = self.qkv(x)              # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # 各 [B, C, H, W]

        # 展平空间：[B, C, HW]
        q = q.reshape(B, C, -1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1)

        # 光谱 attention: [B, C, C]
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = (attn * weight).softmax(dim=-1)
        out = torch.bmm(attn, v)  # [B, C, HW]
        return out.view(B, C, H, W)

    def forward(self, x):
        # 1. 小波分解
        LL, LH, HL, HH = self.dwt2d(x)

        # 2. 各子带独立光谱 attention，权重归一化
        w = torch.softmax(self.band_weights, dim=0)
        LL_attn = self.spectral_attention(LL, w[0])
        LH_attn = self.spectral_attention(LH, w[1])
        HL_attn = self.spectral_attention(HL, w[2])
        HH_attn = self.spectral_attention(HH, w[3])

        # 3. 子带间交互
        bands = torch.cat([LL_attn, LH_attn, HL_attn, HH_attn], dim=1)
        bands = self.band_interact(bands)
        LL_out, LH_out, HL_out, HH_out = bands.chunk(4, dim=1)

        # 4. 逆小波重建
        out = self.idwt2d(LL_out, LH_out, HL_out, HH_out)
        return self.proj(out)


# ──────────────────────────────────────────────
# 统一构建接口
# ──────────────────────────────────────────────

def build_ml_layer(ml_type, dim):
    """根据类型名构建 ML 层

    Args:
        ml_type: 'dwconv_ca' | 'wssa' | 'freq_band'
        dim: 通道数
    Returns:
        nn.Module，forward(x) → x，[B, C, H, W] → [B, C, H, W]
    """
    if ml_type == 'dwconv_ca':
        return ML_DWConvCA(dim)
    elif ml_type == 'wssa':
        return ML_WSSA(dim)
    elif ml_type == 'freq_band':
        return ML_FreqBandAttention(dim)
    else:
        raise ValueError(f"未知 ML 层类型: {ml_type}，可选: dwconv_ca / wssa / freq_band")
