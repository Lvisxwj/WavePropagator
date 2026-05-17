"""
helmholtz_ops.py — 亥姆霍兹频域逆算子 & Beer-Lambert 吸收

被 helm_pure.py (Model 5) 和 wpo3d_helm.py (Model 6) 共用。

数学基础：Helmholtz_HSI_Analysis.md §3（H1-γ）和 §4（H1-β，步骤 B）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HelmholtzInverseOp(nn.Module):
    """
    亥姆霍兹频域逆算子（H1-γ 核心）：

        f_out = IFFT[ FFT(M·s) / (k²(λ) - |ω|² + iε) ]

    输入:
        s    : [B, C, H, W]  源场（已经过编码器）
        mask : [B, C, H, W]  CASSI mask（∈[0,1]）
    输出:
        f    : [B, C, H, W]  亥姆霍兹算子输出（实数）

    可学习参数:
        k_learn [C]   — 波数可学习修正项，初始化为物理值
        gamma_raw     — 软硬先验混合比，sigmoid(gamma_raw)∈(0,1)
        eps_raw [C]   — 共振正则化项，softplus(eps_raw)+1e-6 > 0
    """

    def __init__(self, dim: int, k_init: torch.Tensor):
        """
        Args:
            dim    : 通道数
            k_init : [dim] 物理波数初始值（来自 physics.py）
        """
        super().__init__()
        # 物理波数（固定，不参与梯度）
        self.register_buffer('k_phys', k_init.clone())

        # 可学习修正
        self.k_learn   = nn.Parameter(k_init.clone())
        self.gamma_raw = nn.Parameter(torch.tensor(-2.2))  # sigmoid→0.1：早期物理先验主导

        # 每波段独立的正则化项 ε
        # softplus(-4.6) ≈ 0.01，足够抑制共振奇点
        self.eps_raw = nn.Parameter(torch.full((dim,), -4.6))

    def _get_k_eff(self) -> torch.Tensor:
        gamma = torch.sigmoid(self.gamma_raw)
        return (1.0 - gamma) * self.k_phys + gamma * self.k_learn  # [C]

    def forward(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = s.shape
        device = s.device

        # 1. Mask 源项调制（空间域乘法）
        ms = mask * s  # [B, C, H, W]

        # 2. 2D rFFT
        ms_fft = torch.fft.rfft2(ms)  # [B, C, H, W//2+1]，复数

        # 3. 空间频率网格（弧度/像素）
        fh = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        fw = torch.fft.rfftfreq(W, device=device).view(1, 1, 1, W // 2 + 1)
        omega_sq = (2.0 * math.pi) ** 2 * (fh ** 2 + fw ** 2)  # [1,1,H,W//2+1]

        # 4. 有效波数
        k_eff = self._get_k_eff()               # [C]
        k_sq  = (k_eff ** 2).view(1, C, 1, 1)  # [1,C,1,1]

        # 5. 正则化 ε
        eps = (F.softplus(self.eps_raw) + 1e-6).view(1, C, 1, 1)  # [1,C,1,1]

        # 6. 复数分母：(k² - |ω|²) + iε
        denom_real = (k_sq - omega_sq).expand(1, C, H, W // 2 + 1).contiguous()
        denom_imag = eps.expand(1, C, H, W // 2 + 1).contiguous()
        denom = torch.complex(denom_real, denom_imag)  # [1,C,H,W//2+1]

        # 7. 频域除法
        f_fft = ms_fft / denom  # [B,C,H,W//2+1]

        # 8. iFFT 取实部
        f_out = torch.fft.irfft2(f_fft, s=(H, W))  # [B,C,H,W]

        return f_out


class BeerLambertAbsorption(nn.Module):
    """
    Beer-Lambert 波长依赖吸收修正（H1-β / H2-γ Step 3）：

        f_out = f * exp( -κ₀ · (1 - M) · 2π · L · (1/λ_b_normalized) )

    M 低（mask 遮挡区）→ 吸收强；短波段（1/λ大）→ 吸收强。

    可学习参数：κ₀（消光系数基准），L（等效传播路径）。
    """

    def __init__(self, dim: int, inv_lambda_init: torch.Tensor,
                 init_kappa: float = 0.5, init_L: float = 1.0):
        """
        Args:
            dim             : 通道数
            inv_lambda_init : [dim] 归一化 1/λ（来自 physics.py）
            init_kappa      : κ₀ 的初始值
            init_L          : L 的初始值
        """
        super().__init__()
        self.register_buffer('inv_lambda', inv_lambda_init.clone())

        # softplus 保正：softplus(x) = ln(1+e^x)，反解 x = ln(e^v - 1)
        import numpy as np
        self.kappa_raw = nn.Parameter(
            torch.tensor(float(np.log(np.exp(init_kappa) - 1 + 1e-8))))
        self.L_raw = nn.Parameter(
            torch.tensor(float(np.log(np.exp(init_L) - 1 + 1e-8))))

    def forward(self, f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        kappa = F.softplus(self.kappa_raw)  # 标量
        L     = F.softplus(self.L_raw)      # 标量

        # exponent = -κ₀ · (1-M) · 2π · L · (1/λ_b)
        # inv_lambda: [dim] → [1, C, 1, 1]
        inv_lam  = self.inv_lambda.view(1, -1, 1, 1)
        exponent = -kappa * (1.0 - mask) * (2.0 * math.pi) * L * inv_lam

        # clamp 防止 exp 下溢（≤ 0 的 exponent 才有物理意义）
        exponent = exponent.clamp(min=-30.0, max=0.0)

        return f * torch.exp(exponent)
