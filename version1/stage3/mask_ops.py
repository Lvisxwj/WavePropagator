"""
mask_ops.py — 三种 Mask 添加机制

MaskGateA     — 方案 A：初始振幅软门控（默认，推荐）
MaskSourceB   — 方案 B：Mask 作为源项（非齐次波方程）
MaskKleinGordonD — 方案 D：Klein-Gordon 质量场 + Born 一阶修正

WPO3D 通过 mask_mode='A'/'B'/'D' 选择使用哪种。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGateA(nn.Module):
    """
    方案 A：Mask 作为初始振幅软门控。

    u0 = Phi(x) * gate,  v0 = Psi(x) * gate
    gate = eps + (1-eps) * mask_spatial    (eps=0.1 避免全零)

    对应 CASSI 物理：mask 在编码孔径处对光场做一次性振幅调制。
    """

    def __init__(self, dim, eps=0.1):
        super().__init__()
        self.eps = eps
        # Phi：生成初始场 u0
        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        # Psi：生成速度场 v0
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x, mask_spatial):
        """
        x:            [B, C, H, W]
        mask_spatial: [B, C, H, W]  值域 [0,1]
        返回: u0, v0  各 [B, C, H, W]
        """
        gate = self.eps + (1.0 - self.eps) * mask_spatial
        u0 = self.phi(x) * gate
        v0 = self.psi(x) * gate
        return u0, v0


class MaskSourceB(nn.Module):
    """
    方案 B：Mask 作为源项（Duhamel 原理的离散近似）。

    在 WPO 零阶输出基础上叠加一个 mask 调制的源项贡献：
        correction = IFFT( FFT(mask * S(x)) * sinc_term * decay )
    其中 S 由特征 x 生成。

    注意：sinc_term 和 decay 由 WPO3D 在频域计算后传入。
    """

    def __init__(self, dim):
        super().__init__()
        # 同 MaskGateA 提供标准的 u0/v0（无 mask 门控）
        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        # 源强度生成器
        self.source_net = nn.Conv2d(dim, dim, 1, bias=False)
        self.source_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mask_spatial):
        """返回 u0, v0（无 mask 门控），以及 source 项供 WPO3D 叠加"""
        u0 = self.phi(x)
        v0 = self.psi(x)
        source = mask_spatial * self.source_net(x)
        return u0, v0, source

    def get_source_weight(self):
        return self.source_weight


class MaskKleinGordonD(nn.Module):
    """
    方案 D：Klein-Gordon 质量场 + Born 一阶修正。

    零阶解 = 普通 WPO（与方案 A 相同）
    一阶修正：
        m_sq = m0_sq.clamp(0, 0.5) * (1 - mask_spatial)
        source = -m_sq * u0_out
        correction = IFFT( FFT(source) * sinc_term * decay )
        out = u0_out + kg_weight * correction

    sinc_term, decay 由 WPO3D 计算后传入 apply_correction()。
    """

    def __init__(self, dim, eps=0.1):
        super().__init__()
        self.eps = eps
        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.m0_sq = nn.Parameter(torch.tensor(0.1))
        self.kg_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mask_spatial):
        """返回 u0, v0（带软门控）和质量场 m_sq"""
        gate = self.eps + (1.0 - self.eps) * mask_spatial
        u0 = self.phi(x) * gate
        v0 = self.psi(x) * gate
        m_sq = self.m0_sq.clamp(0.0, 0.5) * (1.0 - mask_spatial)
        return u0, v0, m_sq

    def apply_correction(self, u0_out, m_sq, sinc_term, decay, C, H, W):
        """
        u0_out:    [B, C, H, W]  零阶 WPO 输出（空间域）
        m_sq:      [B, C, H, W]  质量场
        sinc_term: [C, H, W//2+1] complex / real  频域 sinc
        decay:     标量
        返回：修正后的 out [B, C, H, W]
        """
        source = -m_sq * u0_out
        source_fft = torch.fft.rfftn(source, dim=(-3, -2, -1))
        corr_fft = source_fft * sinc_term * decay
        correction = torch.fft.irfftn(corr_fft, s=(C, H, W), dim=(-3, -2, -1))
        return u0_out + self.kg_weight * correction
