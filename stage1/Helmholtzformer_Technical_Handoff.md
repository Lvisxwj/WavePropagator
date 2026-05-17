# Helmholtzformer 技术交接文档 — Claude Code 实现指南

> **目的**：在已完成的 WaveMST 代码框架（4 个 Wave 模型已跑通）基础上，新增 3 个基于亥姆霍兹方程的物理增强模型。本文档专注于**新增内容**，不重复 WaveMST_Technical_Handoff.md 已涵盖的部分（数据加载、CASSI 仿真、训练循环、test/viz 等）。
>
> **前置条件**：用户已实现并正在训练 WaveMST 系列（Model 0–3）。本次新增 Model 4–6，与原有模型共用同一套 dataset/loss/train/test/viz 框架，只需新增模型文件并扩展 train.py 的 MODELS 字典。
>
> **数学基础**：所有数学推导见 `Helmholtz_HSI_Analysis.md`。本文档只给实现细节。

---

## 目录

1. [新增方案概述](#1-新增方案概述)
2. [对原有代码架构的影响](#2-对原有代码架构的影响)
3. [新增文件 1：物理常数与波数预计算 `physics.py`](#3-新增文件-1physics)
4. [新增文件 2：亥姆霍兹算子 `helmholtz_ops.py`](#4-新增文件-2helmholtz_ops)
5. [Model 4 实现：H2-α — `wpo3d_phys.py`](#5-model-4-实现h2-α)
6. [Model 5 实现：H1-γ — `helm_pure.py`](#6-model-5-实现h1-γ)
7. [Model 6 实现：H2-γ — `wpo3d_helm.py`](#7-model-6-实现h2-γ)
8. [train.py 的修改清单](#8-trainpy-的修改清单)
9. [关键陷阱与调试要点](#9-关键陷阱与调试要点)
10. [开发与验证顺序](#10-开发与验证顺序)

---

## 1. 新增方案概述

### 1.1 三个新模型

| 索引 | 名称 | 一句话描述 | 与已有模型的关系 |
|-----|------|----------|---------------|
| 4 | H2-α / WaveMST_Phys | Model 0 (3D-WPO-Pure) 的物理波数版本，把 $v_\lambda\omega_\lambda$ 替换为预计算的 $k_\text{phys}(\lambda)=2\pi/\lambda$ | Model 0 的近亲，只改色散关系 |
| 5 | H1-γ / Helmholtzformer | 纯稳态亥姆霍兹方程，频域逆算子 $1/(k^2-\|\boldsymbol{\omega}\|^2)$，无时间演化 | 全新结构，不依赖 WPO |
| 6 | H2-γ / WaveHelm（主推） | Model 4 + Beer-Lambert 吸收修正 + Mask 双重作用 | Model 4 的扩展版 |

### 1.2 与已有 Wave 系列的核心差异

| 已有 Wave 系列 | 新增 Helmholtz 系列 |
|--------------|------------------|
| 波速 $v_s, v_\lambda$ 都是可学习 | $k_\text{phys}$ 是预计算物理常数（CAVE 28 波段：400–700 nm） |
| 光谱维度做 FFT（$\omega_\lambda$ 是傅里叶频率） | 不对光谱维度做 FFT（$k(\lambda)$ 直接作为标签） |
| 光谱通道是"传播方向" | 光谱通道是"独立振子"，每个有固有频率 |
| 计算 3D rFFT | 仅计算 2D rFFT（空间维），效率略高 |

### 1.3 优先级建议

1. **先实现 Model 4 (H2-α)**：基于 Model 0 修改 30%，最快跑通，验证物理波数注入是否有效
2. **再实现 Model 6 (H2-γ)**：基于 Model 4 加 Beer-Lambert 项，最终主推方案
3. **最后实现 Model 5 (H1-γ)**：作为消融对照（纯稳态 vs 动态）

---

## 2. 对原有代码架构的影响

### 2.1 不需要改动的文件

- `dataset.py`、`mask_ops.py`、`loss.py`、`test.py`、`viz.py` — 全部复用
- `mst.py`、`wpo3d.py`、`wpo_smsa.py`、`wpo_mamba.py` — 已有 4 个 Wave 模型保持不变

### 2.2 需要新增的文件

```
WaveMST/
├── (已有文件...)
├── physics.py            ← 新增：CAVE 波长表、波数预计算、归一化
├── helmholtz_ops.py      ← 新增：亥姆霍兹频域逆算子（Model 5/6 共用）
├── wpo3d_phys.py         ← 新增：Model 4 (H2-α) 模型定义
├── helm_pure.py          ← 新增：Model 5 (H1-γ) 模型定义
└── wpo3d_helm.py         ← 新增：Model 6 (H2-γ) 模型定义
```

### 2.3 需要修改的文件

- `train.py` — 仅扩展 MODELS 字典，新增 3 个索引；其余训练循环逻辑不动

---

## 3. 新增文件 1：`physics.py`

### 3.1 功能

提供 CAVE 数据集的物理波长向量和归一化波数，作为 Model 4/5/6 的物理先验。

### 3.2 设计要点

CAVE 28 波段对应波长（按 MST 论文设定，10 nm 步长）：

```
λ_b ∈ {453, 457, 462, 467, 472, 476, 481, 486, 491, 496, 502, 507,
       515, 526, 537, 547, 558, 569, 580, 590, 600, 611, 622, 633,
       644, 655, 668, 681} nm
```

注意：MST 的 28 波段是从 CAVE 31 波段中**非均匀采样**得到的（短波密集，长波稀疏）。具体波长值参考 `MST/simulation/train_code/utils.py` 或 `vegetation_index/` 中的 wavelength 表。如果找不到，**用线性近似**：从 450 nm 到 680 nm 等差排列 28 个值（误差 ≤5 nm，对模型影响可忽略）。

### 3.3 实现要点

```
class PhysicsParams:
    """提供 CAVE 数据集的物理常数和波数预计算"""
    
    WAVELENGTHS_CAVE = [...]  # 28 个波长（nm），用上面的列表，或线性近似
    
    @staticmethod
    def get_k_phys_normalized(num_bands=28):
        """归一化物理波数 k = 2π/λ，归一化到 [0.6, 1.0] 区间
        返回: tensor [num_bands]
        
        归一化方式：k_tilde = λ_min / λ_b（参考波长选最短）
        这样最短波长 k=1.0，最长波长 k≈0.66
        """
        wavelengths = torch.tensor(WAVELENGTHS_CAVE[:num_bands], dtype=torch.float32)
        lambda_ref = wavelengths.min()
        k_tilde = lambda_ref / wavelengths   # 范围 [~0.66, 1.0]
        return k_tilde
    
    @staticmethod
    def get_inverse_lambda(num_bands=28):
        """返回 1/λ_b（用于 Beer-Lambert 因子的波长依赖项），
        归一化到平均值为 1
        """
        wavelengths = torch.tensor(WAVELENGTHS_CAVE[:num_bands], dtype=torch.float32)
        inv_lambda = 1.0 / wavelengths
        inv_lambda = inv_lambda / inv_lambda.mean()  # 归一化
        return inv_lambda
```

### 3.4 在模型中的使用

```
# 在模型 __init__ 中
self.register_buffer('k_phys', PhysicsParams.get_k_phys_normalized(num_bands))
# k_phys 是常数 buffer，不参与训练，但跟随模型移动设备
```

**注意**：当模型的 dim 不等于 28（即 encoder 中下采样后通道翻倍），$k_\text{phys}$ 需要插值或重新映射。**建议**：物理波数只在最浅层（dim=28）使用；下采样后，让网络用 $k_\text{learn}$ 自由学习（深层特征已不直接对应物理波段）。

---

## 4. 新增文件 2：`helmholtz_ops.py`

### 4.1 功能

实现亥姆霍兹频域逆算子（Model 5/6 共用），以及 Beer-Lambert 吸收因子计算（Model 6 用）。

### 4.2 核心算子设计

```
class HelmholtzInverseOp(nn.Module):
    """亥姆霍兹频域逆算子
    
    计算: f_out = IFFT[ FFT(M·s) / (k²(λ) - |ω|² + iε) ]
    
    输入:
        s:    [B, C, H, W]  源场（已编码）
        mask: [B, C, H, W]  CASSI mask 空间分布
        k_phys: [C]         物理波数（softplus 保正）
    输出:
        f:    [B, C, H, W]  亥姆霍兹算子的输出
    """
    
    def __init__(self, num_bands, learnable_eps=True, learnable_k=True, k_init=None):
        super().__init__()
        # 共振正则化 ε（每波段独立，可学习）
        self.eps_raw = nn.Parameter(torch.full((num_bands,), -4.6))  # softplus(-4.6)≈0.01
        
        # 可学习波数修正
        if learnable_k:
            self.k_learn = nn.Parameter(k_init.clone())  # 初始化为物理值
            self.gamma_raw = nn.Parameter(torch.tensor(-2.2))  # sigmoid(-2.2)≈0.1
        else:
            self.register_buffer('k_learn', k_init)
            self.gamma_raw = None
    
    def get_k_eff(self, k_phys):
        """软硬先验混合"""
        if self.gamma_raw is None:
            return k_phys
        gamma = torch.sigmoid(self.gamma_raw)
        return (1 - gamma) * k_phys + gamma * self.k_learn
    
    def forward(self, s, mask, k_phys):
        B, C, H, W = s.shape
        device = s.device
        
        # 1. Mask 源项调制（空间域乘法）
        ms = mask * s   # [B, C, H, W]
        
        # 2. 2D rFFT
        ms_fft = torch.fft.rfft2(ms, dim=(-2, -1))  # [B, C, H, W//2+1]
        
        # 3. 构建空间频率网格
        fh = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        fw = torch.fft.rfftfreq(W, device=device).view(1, 1, 1, W // 2 + 1)
        omega_sq = (2 * 3.14159265) ** 2 * (fh ** 2 + fw ** 2)  # [1, 1, H, W//2+1]
        
        # 4. 有效波数（混合先验）
        k_eff = self.get_k_eff(k_phys)         # [C]
        k_sq = (k_eff ** 2).view(1, C, 1, 1)   # [1, C, 1, 1]
        
        # 5. 正则化项 ε
        eps = F.softplus(self.eps_raw).view(1, C, 1, 1) + 1e-6
        
        # 6. 复数分母 + 频域除法
        denom = (k_sq - omega_sq) + 1j * eps   # [1, C, H, W//2+1]，complex
        f_fft = ms_fft / denom
        
        # 7. iFFT 取实部
        f_out = torch.fft.irfft2(f_fft, s=(H, W), dim=(-2, -1))
        
        return f_out


class BeerLambertAbsorption(nn.Module):
    """Beer-Lambert 吸收因子: exp(-κ₀ · (1-M) · 2πL/λ_b)
    
    输入:
        f:        [B, C, H, W]  传播后的场
        mask:     [B, C, H, W]  CASSI mask
        inv_lambda: [C]         波长倒数（归一化）
    输出:
        f * exp(-...)
    """
    
    def __init__(self, num_bands, init_kappa=0.5, init_L=1.0):
        super().__init__()
        # softplus 保证正值
        self.kappa_raw = nn.Parameter(torch.tensor(np.log(np.exp(init_kappa) - 1)))
        self.L_raw = nn.Parameter(torch.tensor(np.log(np.exp(init_L) - 1)))
    
    def forward(self, f, mask, inv_lambda):
        kappa = F.softplus(self.kappa_raw)
        L = F.softplus(self.L_raw)
        
        # exponent: [B, C, H, W]
        # κ₀ * (1-M) * 2πL/λ_b
        # inv_lambda 已归一化为均值 1，扮演"波长依赖系数"角色
        inv_lam = inv_lambda.view(1, -1, 1, 1)
        exponent = -kappa * (1 - mask) * 2 * 3.14159265 * L * inv_lam
        
        # 数值稳定：clamp 避免 exp 爆炸或下溢
        exponent = exponent.clamp(min=-30, max=0)
        
        return f * torch.exp(exponent)
```

### 4.3 数值稳定性要点

- `eps_raw` 初始化为 -4.6，softplus 后约 0.01（标准共振正则化值）
- `gamma_raw` 初始化为 -2.2，sigmoid 后约 0.1（早期物理先验主导）
- Beer-Lambert exponent 必须 clamp，避免 mask=0 处 exp(-large) 下溢
- 频域分母用 complex（PyTorch 支持），不要写成 |k²-ω²|+ε（这会丢失相位）

---

## 5. Model 4 实现：H2-α — `wpo3d_phys.py`

### 5.1 与 Model 0 (`wpo3d.py`) 的差异

| 项目 | Model 0 (3D-WPO-Pure) | Model 4 (H2-α) |
|-----|---------------------|---------------|
| FFT 维度 | 3D rFFT (C, H, W) | 2D rFFT (H, W)，逐通道 |
| 色散关系 | $v_s^2(\omega_x^2+\omega_y^2)+v_\lambda^2\omega_\lambda^2$ | $v_s^2(\omega_x^2+\omega_y^2)+k_\text{eff}^2(\lambda)$ |
| 光谱维 | 作为传播方向（FFT） | 作为振子标签（每个通道有固有频率） |
| 可学习参数 | $v_s, v_\lambda, \alpha, t$ | $v_s, \alpha, t, \gamma, k_\text{learn}$ |

### 5.2 WPO3D_Phys 模块

```
class WPO3D_Phys(nn.Module):
    """物理波数注入的 WPO 模块（H2-α）
    
    每个通道是独立振子，固有频率由物理波数决定
    """
    
    def __init__(self, dim, num_bands_for_kphys=28):
        super().__init__()
        # 可学习 WPO 参数
        self.alpha_raw = nn.Parameter(torch.tensor(np.log(np.exp(0.1) - 1)))  # softplus → 0.1
        self.vs_raw = nn.Parameter(torch.tensor(np.log(np.exp(1.0) - 1)))     # softplus → 1.0
        self.t_raw = nn.Parameter(torch.tensor(np.log(np.exp(1.0) - 1)))      # softplus → 1.0
        
        # 软硬先验混合
        self.gamma_raw = nn.Parameter(torch.tensor(-2.2))  # sigmoid → 0.1
        
        # 可学习波数修正（初始化为物理值）
        k_init = PhysicsParams.get_k_phys_normalized(dim)
        if dim != num_bands_for_kphys:
            # 深层特征通道数 ≠ 28，用线性插值
            k_init = F.interpolate(k_init.view(1,1,-1), size=dim, mode='linear', align_corners=True).view(-1)
        self.register_buffer('k_phys', k_init)            # 不可学习的物理值
        self.k_learn = nn.Parameter(k_init.clone())      # 可学习修正项
        
        # 语义编码器 Φ 和速度编码器 Ψ
        self.phi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # DWConv
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        # SiLU gate（参照 WaveFormer Wave2D 设计）
        self.gate_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)
    
    def get_k_eff(self):
        gamma = torch.sigmoid(self.gamma_raw)
        return (1 - gamma) * self.k_phys + gamma * self.k_learn
    
    def forward(self, x, mask_spatial, mask_eps=0.1):
        B, C, H, W = x.shape
        device = x.device
        
        # 1. Mask 软门控（方案 A）
        gate = mask_eps + (1 - mask_eps) * mask_spatial   # [B, C, H, W]
        u0 = self.phi(x) * gate
        v0 = self.psi(x) * gate
        
        # SiLU gate 输入（在 FFT 前 split）
        z = self.gate_proj(x)
        
        # 2. 2D rFFT（仅空间维）
        u0_fft = torch.fft.rfft2(u0, dim=(-2, -1))   # [B, C, H, W//2+1]
        v0_fft = torch.fft.rfft2(v0, dim=(-2, -1))
        
        # 3. 空间频率网格
        fh = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)
        fw = torch.fft.rfftfreq(W, device=device).view(1, 1, 1, W // 2 + 1)
        omega_xy_sq = (2 * 3.14159265) ** 2 * (fh ** 2 + fw ** 2)   # [1, 1, H, W//2+1]
        
        # 4. 物理波数 k_eff(λ)
        k_eff = self.get_k_eff()                  # [C]
        k_sq = (k_eff ** 2).view(1, C, 1, 1)      # [1, C, 1, 1]
        
        # 5. WPO 参数
        alpha = F.softplus(self.alpha_raw)
        vs = F.softplus(self.vs_raw)
        t = F.softplus(self.t_raw)
        
        # 6. 色散关系 ω₀² 和判别式 η
        omega0_sq = vs ** 2 * omega_xy_sq + k_sq  # [1, C, H, W//2+1]
        eta = omega0_sq - (alpha / 2) ** 2
        
        # 7. 欠阻尼/过阻尼分区处理（关键！）
        underdamped = (eta > 1e-8)
        overdamped = (eta < -1e-8)
        critical = ~underdamped & ~overdamped
        
        sqrt_pos = torch.sqrt(eta.clamp(min=1e-8))    # ω_d
        sqrt_neg = torch.sqrt((-eta).clamp(min=1e-8))  # γ
        
        # Cs(η, t): cos(ω_d t) 或 cosh(γ t)
        cos_term = torch.where(underdamped,
                               torch.cos(sqrt_pos * t),
                               torch.cosh(sqrt_neg * t.clamp(max=10)))  # 防止 cosh 爆炸
        cos_term = torch.where(critical, torch.ones_like(cos_term), cos_term)
        
        # Sn(η, t): sin/ω_d 或 sinh/γ
        sin_term = torch.where(underdamped,
                               torch.sin(sqrt_pos * t) / (sqrt_pos + 1e-8),
                               torch.sinh((sqrt_neg * t).clamp(max=10)) / (sqrt_neg + 1e-8))
        sin_term = torch.where(critical, t * torch.ones_like(sin_term), sin_term)
        
        # 8. 闭式解
        decay = torch.exp(-alpha * t / 2)
        out_fft = decay * (u0_fft * cos_term + (v0_fft + alpha / 2 * u0_fft) * sin_term)
        
        # 9. iFFT
        u_out = torch.fft.irfft2(out_fft, s=(H, W), dim=(-2, -1))   # [B, C, H, W]
        
        # 10. SiLU gate + 输出投影
        out = u_out * F.silu(z)
        out = self.out_proj(out)
        
        return out


class WPO3DPhysBlock(nn.Module):
    """完整 Block: LN → WPO3D_Phys → Residual → LN → FFN → Residual"""
    
    def __init__(self, dim, ffn_expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.wpo = WPO3D_Phys(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expand)  # 复用 mst.py 中的 FeedForward
    
    def forward(self, x, mask):
        # x: [B, C, H, W], mask: [B, C, H, W]
        # LayerNorm 在通道维度，需要 permute
        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm1(x_perm).permute(0, 3, 1, 2)
        x = x + self.wpo(x_norm, mask)
        
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_perm).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm)
        
        return x


class WaveMST_Phys(nn.Module):
    """完整模型（Model 4），U-Net 结构与 Model 0 相同
    
    构造、forward 完全复制 wpo3d.py 的 WaveMST_3D，仅把内部 Block 替换为 WPO3DPhysBlock
    """
    
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2]):
        super().__init__()
        # ... 复制 WaveMST_3D 的 __init__
        # 把 WPO3DBlock 替换为 WPO3DPhysBlock
        # 注意每个 stage 内 dim 是不同的
    
    def forward(self, x, input_mask):
        # 与 WaveMST_3D 的 forward 完全相同
        # mask 处理：从 shifted mask 截取 256 宽度，得到 spatial mask
        ...
```

### 5.3 关键陷阱

**陷阱 1：dim 翻倍后 k_phys 怎么办？**

Encoder 第二层 dim=56，超过 28 个物理波段。两种解决方案：

- **方案 A**（推荐）：物理波数只在第一层用，深层让 $k_\text{learn}$ 自由学习。即第二层及以后 `gamma=1`（完全可学习），第一层 `gamma=0.1`。
- **方案 B**：用 `F.interpolate` 把 28 个物理值线性插值到任意 dim 数。简单但物理意义弱。

代码里建议先用方案 B 跑通，再用方案 A 做消融。

**陷阱 2：cosh 数值爆炸**

过阻尼区域 `cosh(γt)` 会指数增长，虽然外部 `decay=exp(-αt/2)` 会压制，但中间计算可能溢出。

解决：限制 `(sqrt_neg * t)` 的最大值（如 clamp(max=10)），或者把 decay 直接合并进 cos/sin 计算（更稳定）。

---

## 6. Model 5 实现：H1-γ — `helm_pure.py`

### 6.1 模型设计哲学

Model 5 是**纯稳态**模型——没有时间演化（没有 $t$ 参数），整个网络是亥姆霍兹算子的级联。每一层做一次 "源场编码 → mask 调制 → 亥姆霍兹逆算子" 的循环。

可以理解为：把 Model 0 的 WPO 整个换成 HelmholtzInverseOp，FFN 保留。

### 6.2 实现要点

```
class HelmBlock(nn.Module):
    """亥姆霍兹 Block: LN → HelmholtzInverse → Residual → LN → FFN → Residual"""
    
    def __init__(self, dim, num_bands_for_kphys=28):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # 源场编码器（参照 Wave 模型的 Φ 设计）
        self.source_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        # 物理波数 buffer
        k_init = PhysicsParams.get_k_phys_normalized(dim)
        if dim != num_bands_for_kphys:
            k_init = F.interpolate(k_init.view(1,1,-1), size=dim, mode='linear', align_corners=True).view(-1)
        self.register_buffer('k_phys', k_init)
        
        # 亥姆霍兹算子
        self.helm_op = HelmholtzInverseOp(dim, k_init=k_init)
        
        # 输出投影
        self.gate_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, 2)
    
    def forward(self, x, mask):
        # x: [B, C, H, W]
        # 1. LN
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_perm).permute(0, 3, 1, 2)
        
        # 2. 源场编码
        s = self.source_encoder(x_norm)
        
        # 3. SiLU gate input
        z = self.gate_proj(x_norm)
        
        # 4. 亥姆霍兹算子: f = IFFT[ FFT(M·s) / (k² - |ω|² + iε) ]
        f = self.helm_op(s, mask, self.k_phys)
        
        # 5. SiLU gate
        out = f * F.silu(z)
        out = self.out_proj(out)
        
        # 6. Residual
        x = x + out
        
        # 7. FFN
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_perm).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm)
        
        return x


class Helmholtzformer(nn.Module):
    """Model 5: 纯稳态亥姆霍兹模型
    
    U-Net 结构与 Model 0 相同，每层用 HelmBlock 替代 WPO3DBlock
    """
    
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2]):
        super().__init__()
        self.stage = stage
        
        # Embedding
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                nn.ModuleList([HelmBlock(dim_stage) for _ in range(num_blocks[i])]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),  # FeaDownSample
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),  # MaskDownSample
            ]))
            dim_stage *= 2
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([HelmBlock(dim_stage) for _ in range(num_blocks[-1])])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),  # Fusion
                nn.ModuleList([HelmBlock(dim_stage // 2) for _ in range(num_blocks[stage - 1 - i])]),
            ]))
            dim_stage //= 2
        
        # Mapping
        self.mapping = nn.Conv2d(dim, 28, 3, 1, 1)
    
    def forward(self, x, input_mask):
        # x: [B, 28, 256, 256], input_mask: [B, 28, 256, 310] (shifted)
        # 转回 spatial mask
        mask_spatial = shift_back(input_mask, step=2)  # [B, 28, 256, 256]
        
        fea = self.embedding(x)
        mask = mask_spatial
        
        # Encoder
        fea_encoder = []
        masks = []
        for (blocks, fea_down, mask_down) in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask)
            fea_encoder.append(fea)
            masks.append(mask)
            fea = fea_down(fea)
            mask = mask_down(mask)
        
        # Bottleneck
        for blk in self.bottleneck:
            fea = blk(fea, mask)
        
        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask)
        
        out = self.mapping(fea) + x
        return out
```

### 6.3 注意事项

- 因为没有"时间步"概念，HelmBlock 内部不需要 alpha, t 等 WPO 参数
- 模型容量比 Model 4 略小（少了 4 个 WPO 标量参数 × 层数）
- 训练时若性能明显低于 Model 4，说明动态传播比稳态共振更重要——这正是消融实验要回答的问题

---

## 7. Model 6 实现：H2-γ — `wpo3d_helm.py`

### 7.1 与 Model 4 的关系

Model 6 = Model 4 + Beer-Lambert 吸收修正（在每个 Block 末尾追加）。

### 7.2 实现策略

```
class WPO3DHelmBlock(nn.Module):
    """H2-γ Block: 在 WPO3DPhysBlock 基础上加 Beer-Lambert 吸收"""
    
    def __init__(self, dim, num_bands_for_kphys=28):
        super().__init__()
        # 复用 Model 4 的 Block
        self.wpo_block = WPO3DPhysBlock(dim)
        
        # 新增 Beer-Lambert 吸收层
        # 每个 Block 有自己的 κ₀, L 参数
        inv_lambda_init = PhysicsParams.get_inverse_lambda(dim)
        if dim != num_bands_for_kphys:
            inv_lambda_init = F.interpolate(inv_lambda_init.view(1,1,-1), size=dim, mode='linear').view(-1)
        self.register_buffer('inv_lambda', inv_lambda_init)
        
        self.absorption = BeerLambertAbsorption(dim, init_kappa=0.5, init_L=1.0)
    
    def forward(self, x, mask):
        # 1. 标准 WPO Phys Block（含 Step 1 初始门控 + Step 2 物理波数 WPO）
        x = self.wpo_block(x, mask)
        
        # 2. Step 3: Beer-Lambert 空间吸收
        x = self.absorption(x, mask, self.inv_lambda)
        
        return x


class WaveMST_Helm(nn.Module):
    """Model 6: 三合一 H2-γ 主推方案
    
    U-Net 结构同 Model 4，每个 Block 替换为 WPO3DHelmBlock
    """
    
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2]):
        super().__init__()
        # 同 WaveMST_Phys 的 __init__，但 Block 用 WPO3DHelmBlock
        # ...
    
    def forward(self, x, input_mask):
        # 同 WaveMST_Phys.forward
        # ...
```

### 7.3 关键设计选择

**选择 1：吸收项放在 Block 内还是 Block 外？**

Block 内：每个 Block 都做一次吸收（推荐，物理意义强：每个传播阶段都有吸收）。
Block 外：只在 U-Net 输出做一次吸收（计算量小，但物理意义弱）。

我们采用 Block 内方案。

**选择 2：吸收项放在 WPO 之前还是之后？**

WPO 之前：`f → 吸收 → WPO → 残差`（Strang 分裂）
WPO 之后：`f → WPO → 吸收 → 残差`（Lie 分裂）

理论上 Strang 二阶精度更好，但实现稍复杂。先用 Lie 分裂（WPO 之后），简单且足够。

**选择 3：吸收因子 inv_lambda 是否每层不同？**

理论上每层应该有不同的"等效波长"（因为 dim 翻倍后通道含义改变）。
实现简单起见：每个 Block 用预计算的 inv_lambda（深层用插值版本），让 $\kappa_0, L$ 学习剩余差异。

---

## 8. train.py 的修改清单

只修改 MODELS 字典和 build_model 函数，**其他训练逻辑完全不动**。

### 8.1 MODELS 字典扩展

```
MODELS = {
    # 已有 4 个
    0: ('WaveMST_3D',       '3d_wpo_pure'),
    1: ('WaveMST_KG',       '3d_wpo_kg'),
    2: ('WaveMST_Parallel', '3d_wpo_smsa'),
    3: ('WaveMST_Mamba',    '2d_wpo_mamba'),
    
    # 新增 3 个
    4: ('WaveMST_Phys',     'h2_alpha_phys'),       # H2-α
    5: ('Helmholtzformer',  'h1_gamma_helm_pure'),  # H1-γ
    6: ('WaveMST_Helm',     'h2_gamma_main'),       # H2-γ (主推)
}
```

### 8.2 build_model 函数扩展

```
def build_model(index):
    if index == 0:
        from wpo3d import WaveMST_3D
        return WaveMST_3D(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, mask_mode=MASK_MODE)
    elif index == 1:
        from wpo3d import WaveMST_3D
        return WaveMST_3D(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, mask_mode=MASK_MODE, use_kg=True)
    elif index == 2:
        from wpo_smsa import WaveMST_Parallel
        return WaveMST_Parallel(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, mask_mode=MASK_MODE)
    elif index == 3:
        from wpo_mamba import WaveMST_Mamba
        return WaveMST_Mamba(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, mask_mode=MASK_MODE)
    
    # 新增
    elif index == 4:
        from wpo3d_phys import WaveMST_Phys
        return WaveMST_Phys(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS)
    elif index == 5:
        from helm_pure import Helmholtzformer
        return Helmholtzformer(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS)
    elif index == 6:
        from wpo3d_helm import WaveMST_Helm
        return WaveMST_Helm(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS)
    else:
        raise ValueError(f"Unknown model index: {index}")
```

### 8.3 CONFIG 区域扩展（可选）

如果希望对新模型用不同的训练超参（比如 H2-γ 因为参数多需要更小学习率），可以加一个分支：

```
# CONFIG
MODEL_INDEX = 6   # 主推 H2-γ

# 学习率自适应（可选）
if MODEL_INDEX in [4, 5, 6]:
    LEARNING_RATE = 3e-4   # 物理增强模型用稍小学习率，初期更稳定
else:
    LEARNING_RATE = 4e-4   # 已有模型保持原值
```

---

## 9. 关键陷阱与调试要点

### 9.1 物理波数的尺度问题（最重要！）

`PhysicsParams.get_k_phys_normalized` 返回值在 [0.66, 1.0] 区间。但 WPO 的色散关系是 $v_s^2|\boldsymbol{\omega}|^2 + k^2$，需要确保两项尺度匹配。

空间频率网格 $|\boldsymbol{\omega}|^2 = (2\pi)^2(\text{fftfreq}^2 + \text{rfftfreq}^2)$，在 $H=W=256$ 时最大值约 $(2\pi \times 0.5)^2 \approx 9.87$。

如果 $k^2 \in [0.43, 1.0]$，则 $k^2 \ll v_s^2|\boldsymbol{\omega}|^2$（$v_s$ 初始化 1.0 时），物理波数项的影响很小。

**解决方案**：让 $k_\text{phys}$ 的归一化与空间频率匹配，比如归一化到 $k \in [\pi, 2\pi]$（让 $k^2$ 与 $|\boldsymbol{\omega}|^2_\text{max}$ 同数量级）：

```
@staticmethod
def get_k_phys_normalized(num_bands=28, target_scale=2 * np.pi):
    wavelengths = torch.tensor(WAVELENGTHS_CAVE[:num_bands], dtype=torch.float32)
    lambda_min = wavelengths.min()
    k_tilde = lambda_min / wavelengths   # [0.66, 1.0]
    return k_tilde * target_scale         # [4.18, 6.28]
```

跑通之前先用 `print(k_phys.min(), k_phys.max(), omega_xy_sq.max())` 确认尺度匹配。

### 9.2 频域分母为零的奇点

H1-γ 的分母 `k² - |ω|² + iε`，当 `|ω| ≈ k` 时实部接近零，仅由 `iε` 支撑。如果 `ε` 太小（< 1e-4），分母数值不稳定。

**调试**：在第一次 forward 后打印 `denom.abs().min()`，确认 ≥ 1e-3。如果太小，提高 `eps_raw` 初始化。

### 9.3 复数 tensor 的 backward

PyTorch 1.8+ 支持复数 tensor 的自动微分，但有些操作（如 `complex.real / 0`）会产生 NaN 梯度。

**调试**：`with torch.autograd.detect_anomaly(): pred = model(...)` 在前几次 iteration 用，确认没有 NaN。

### 9.4 Mask 下采样后值域漂移

Conv2d 下采样的 mask 值域可能超出 [0, 1]，影响 Beer-Lambert 项 (1-M) 的物理意义。

**解决**：每次下采样后做 `mask = mask.clamp(0, 1)` 或 `mask = torch.sigmoid(mask)`。

### 9.5 物理波数与可学习波数的混合

`gamma` 初始化为 sigmoid(-2.2)≈0.1，前期物理先验主导。但如果训练发散，可能是 `k_learn` 偏离物理值太远。

**调试**：定期打印 `(k_learn - k_phys).abs().mean()`，确认偏离量稳定增长而不是爆炸。如果爆炸，给 `k_learn` 加 weight decay。

### 9.6 与 Model 0 的输出量级对比

跑通后做 sanity check：在同一输入下，Model 4 的输出量级应与 Model 0 接近（差异在 2 倍以内）。如果差异极大，多半是物理波数尺度问题（陷阱 9.1）。

---

## 10. 开发与验证顺序

### 10.1 推荐顺序

```
Day 1-2: physics.py + helmholtz_ops.py
  ├── 单元测试: 在小 tensor 上验证算子输出形状、梯度可传
  └── Sanity: HelmholtzInverseOp(随机输入) 输出应是有限值（无 NaN/Inf）

Day 3-4: wpo3d_phys.py (Model 4)
  ├── 复制 wpo3d.py，逐步替换 3D FFT 为 2D FFT
  ├── 添加 k_phys buffer 和 k_learn 参数
  ├── 在 1 个 batch 上 forward+backward 跑通
  └── 跑 5 epoch 小数据集，loss 应下降

Day 5: train.py 扩展
  ├── 添加 Model 4 到 MODELS 和 build_model
  ├── 跑 30 epoch（约 4 小时），观察 PSNR 是否接近 Model 0
  └── 记录基线 PSNR

Day 6-7: wpo3d_helm.py (Model 6, 主推)
  ├── 在 Model 4 基础上添加 BeerLambertAbsorption
  ├── 训练 30 epoch，PSNR 应高于 Model 4
  └── 如果性能不升反降，检查 κ₀ 初始化（可能太大）

Day 8: helm_pure.py (Model 5)
  ├── 实现 HelmBlock 和 Helmholtzformer
  ├── 训练 30 epoch
  └── 用作 H2-γ 的消融对照（动态 vs 稳态）

Day 9-10: 完整训练 + 消融实验
  ├── Model 0/4/5/6 各自训练 200 epoch
  ├── 记录 PSNR/SSIM/SAM 对比
  └── 写论文 ablation table
```

### 10.2 验证检查表

每实现一个模型，按顺序检查：

- [ ] 模型实例化成功，无错误
- [ ] 在 batch_size=1 的随机输入上 forward 成功
- [ ] 输出 shape 正确（[B, 28, 256, 256]）
- [ ] backward 成功，无 NaN/Inf 梯度
- [ ] 一个完整 epoch 的训练 loss 下降
- [ ] 在测试集上 PSNR > 25 dB（10 epoch 后的下限）
- [ ] 在测试集上 PSNR > 30 dB（50 epoch 后）
- [ ] 在测试集上 PSNR > 33 dB（200 epoch 后）

如果某一项不满足，按 §9 的陷阱列表排查。

### 10.3 论文实验表（最终目标）

| Model | PSNR | SSIM | SAM | Params (M) | FLOPs (G) |
|-------|------|------|-----|-----------|----------|
| MST (baseline) | 33.0 | 0.92 | 0.10 | 2.0 | 12.0 |
| Model 0 (3D-WPO) | 33.5? | ? | ? | 2.0 | 11.5 |
| Model 4 (H2-α) | 34.0? | ? | ? | 2.05 | 11.0 |
| Model 5 (H1-γ) | 33.2? | ? | ? | 1.9 | 10.0 |
| **Model 6 (H2-γ, ours)** | **34.5?** | **?** | **?** | 2.1 | 11.5 |

? 是预期值，实际由实验决定。如果 Model 6 不显著优于 Model 4，需要回到方法论检查 Beer-Lambert 项的实现。

---

## 附录：与 Helmholtz_HSI_Analysis.md 的对应关系

| 数学文档（Helmholtz_HSI_Analysis.md）| 本文档（实现） |
|-----------------------------------|-------------|
| §3 H1-γ 推导 | §6 helm_pure.py，HelmholtzInverseOp |
| §4 H1-β（暂不实现） | — |
| §5 H2-α 推导 | §5 wpo3d_phys.py，WPO3D_Phys |
| §6 H2-γ 推导（主推）| §7 wpo3d_helm.py，WPO3DHelmBlock |
| §6.2 公式 (6.7) | §7 forward 代码（WPO + 吸收） |

---

## 附录：与已有 4 个模型的关系矩阵

| 已有 | 新增 | 关系 |
|-----|-----|------|
| Model 0 (3D-WPO) | Model 4 (H2-α) | 4 = 0 + 物理波数替代可学习 v_λ |
| Model 0 | Model 5 (H1-γ) | 5 = 把 0 的 WPO 换成稳态亥姆霍兹 |
| Model 4 | Model 6 (H2-γ) | 6 = 4 + Beer-Lambert 吸收 |
| Model 1 (KG) | Model 4 | 1 用可学习质量场，4 用物理波数（KG 的物理约束版） |

