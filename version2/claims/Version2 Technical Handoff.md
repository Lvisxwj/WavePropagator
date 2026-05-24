# Version 2 技术交接：退化估计 + WPO Propagator 框架

> **目标**：在 version1/stage2 的 3D-WPO unfolding 基础上（5stg 38.21 dB），引入退化估计、FBGW 频带加权、A-HQS 动量、Swin 窗口 WPO、局部精化，预期达到 39+ dB。
> 
> **代码基础**：从 `version1/stage2/` 复制核心文件到 `version2/`，重组目录结构。version1 的 stage4（ML 层实验）已否决，不复制。
> 
> **关键原则**：WPO3D 核心闭式解不动，只在外围增加退化估计和轻量增强。

-----

## 目录

1. [目录结构与文件来源](#1-目录结构与文件来源)
1. [config.yaml 完整内容](#2-configyaml-完整内容)
1. [__init__.py 完整内容](#3-__init__py-完整内容)
1. [新增文件：model/degradation.py](#4-新增文件modeldegradationpy)
1. [新增文件：model/refinement.py](#5-新增文件modelrefinementpy)
1. [修改文件：model/wpo3d.py](#6-修改文件modelwpo3dpy)
1. [修改文件：model/unfolding.py](#7-修改文件modelunfoldingpy)
1. [修改文件：train.py](#8-修改文件trainpy)
1. [修改文件：test.py](#9-修改文件testpy)
1. [修改文件：dataset.py](#10-修改文件datasetpy)
1. [关键陷阱](#11-关键陷阱)
1. [开发顺序与验证](#12-开发顺序与验证)

-----

## 1. 目录结构与文件来源

### 1.1 最终目录

```
CASSI/version2/
├── config.yaml           # 所有超参数（GPU, batch, LR, epochs, paths, etc.）
├── __init__.py           # 模型选择开关 + unfolding 配置 + checkpoint 路径
├── train.py              # 训练入口（只读 config.yaml 和 __init__.py）
├── test.py               # 测试入口
├── dataset.py            # 数据加载
└── model/                # 所有模型组件
    ├── wpo3d.py           # WPO3D 核心 + WPO3DBlock + WaveMST_3D（修改：+FBGW +SwinWPO）
    ├── degradation.py     # DegradationEstimation 模块（新增）
    ├── unfolding.py       # A-HQS unfolding wrapper（修改：+动量 +退化估计接口）
    ├── refinement.py      # LocalRefinement 模块（新增）
    ├── mask_ops.py        # MaskGateA, MaskKleinGordonD（复制，删除 MaskSourceB）
    └── utils.py           # shift/shift_back, ParaEstimator, compute_PhiPhiT（复制）
```

### 1.2 文件来源映射

```bash
# 第一步：创建目录
mkdir -p CASSI/version2/model

# 第二步：从 version1/stage2 复制（这些是经过验证的代码）
cp version1/stage2/dataset.py      version2/dataset.py
cp version1/stage2/mask_ops.py     version2/model/mask_ops.py
cp version1/stage2/unfolding_ops.py version2/model/utils.py      # 重命名
cp version1/stage2/wpo3d.py        version2/model/wpo3d.py
cp version1/stage2/wpo3d_unfold.py version2/model/unfolding.py   # 重命名

# 第三步：新建文件
touch version2/config.yaml
touch version2/__init__.py
touch version2/model/degradation.py
touch version2/model/refinement.py

# 第四步：基于 version1/stage2 的 train.py/test.py 重写
# （不复制——结构大改，从零写更清晰）
```

### 1.3 不复制的文件

- `version1/stage4/ml_layers.py` — ML 层堆砌已否决
- `version1/stage4/enhancement_ops.py` — 色散修正已否决
- `version1/stage2/physics.py` — 不再需要
- `version1/stage2/loss.py` — 简单到不需要单独文件，内联到 train.py

-----

## 2. config.yaml 完整内容

```yaml
# ════════════════════════════════════════════
# config.yaml — 所有训练/测试超参数
# ════════════════════════════════════════════

# GPU & 训练
gpu_id: '0'
batch_size: 5
max_epoch: 300
learning_rate: 4.0e-4
scheduler: 'CosineAnnealingLR'
epoch_sample: 5000
save_thresh: 28.0
use_amp: false

# 数据
crop_size: 256
num_bands: 28
input_setting: 'H'

# 数据路径（绝对路径，永远不变）
data_root: '/data5/SCI/xieweijie/CASSI/dataset'
train_path: '/data5/SCI/xieweijie/CASSI/dataset/CAVE_1024_npy'
test_path: '/data5/SCI/xieweijie/CASSI/dataset/TSA_simu_data/Truth'
mask_path: '/data5/SCI/xieweijie/CASSI/dataset/TSA_simu_data'

# 模型结构（传给 WaveMST_3D）
dim: 28
unet_stage: 3        # U-Net encoder 层数
num_blocks: [2, 2, 2]
```

-----

## 3. __init__.py 完整内容

```python
"""
__init__.py — 模型选择开关 + Unfolding 配置 + Checkpoint 路径

train.py 和 test.py 共享此文件。修改这里的开关即可切换模型配置。
"""

# ── 模型核心选择 ──
USE_KG = False                     # True → KG 方程（mask_mode='D'），False → 纯 WPO（mask_mode='A'）

# ── WPO FBGW 频带引导加权 ──
WPO_FBGW_MODE = 'none'            # 'none' / 'snr_adaptive' / 'learnable_band'

# ── Swin-WPO 窗口传播 ──
USE_SWIN_WPO = False               # True → 64×64 窗内传播 + shift window
SWIN_WINDOW_SIZE = 64              # 窗大小（不能小于 56 = CASSI shift 跨度）

# ── Unfolding 配置 ──
USE_UNFOLDING = True               # True → deep unfolding，False → 端到端
NUM_STAGES = 5                     # unfolding stage 数
SHARE_STAGE_WEIGHTS = True         # True → 所有 stage 共享 prior 权重
MULTI_STAGE_LOSS = True            # True → DPU 风格多 stage 加权损失

# ── Checkpoint（test.py 使用）──
BEST_CKPT = ''                     # 训练完成后填入 best.pth 路径
```

-----

## 4. 新增文件：model/degradation.py

### 4.1 完整代码

```python
"""
degradation.py — 三合一退化估计模块

同时输出：
  1. delta_Phi  [B, C, H, W]  — sensing error 修正（用于 GD step）
  2. deg_weight [B, C, H, W]  — 空间退化权重（净化 WPO 初始场）
  3. sigma      [B, 1, 1, 1]  — 噪声水平（控制 WPO 阻尼 α_eff = α + λσ）

参考：
  - DPU (CVPR 2024) 的 DPB：退化 mask → 1×1 Conv → Sigmoid → 权重
  - DERNN-LNLT (2024) 的 DEN：残差学习估计 sensing error + noise level
"""

import torch
import torch.nn as nn


def construct_degraded_mask(Phi, len_shift=2):
    """构造退化 mask Φ*：shift → compress → reverse。
    
    参考 DPU/Model.py 的 reverse 操作（MST initialization）。
    
    Phi: [B, C, H, W] spatial mask
    返回: [B, C, H, W] 退化 mask（包含 shift+compression 退化信息）
    """
    B, C, H, W = Phi.shape
    # shift each band
    shifted = torch.zeros(B, C, H, W + (C - 1) * len_shift,
                          device=Phi.device, dtype=Phi.dtype)
    for c in range(C):
        shifted[:, c, :, c * len_shift: c * len_shift + W] = Phi[:, c, :, :]
    
    # compress (sum along spectral dim) and broadcast back
    compressed = shifted.sum(dim=1, keepdim=True)  # [B, 1, H, W']
    
    # reverse: assign back to each band
    Phi_star = torch.zeros_like(Phi)
    for c in range(C):
        Phi_star[:, c, :, :] = compressed[:, 0, :, c * len_shift: c * len_shift + W]
    
    # normalize
    Phi_star = 2.0 * Phi_star / C
    
    return Phi_star


class DegradationEstimation(nn.Module):
    """三合一退化估计
    
    参数量（dim=28, hidden=32）：
      delta_phi:  2 × 28 × 28 = 1568
      deg_weight: 56 × 32 + 32 × 28 = 2688
      sigma_est:  28 × 32 + 32 × 1 = 928
      总计: ~5.2K 参数
    """
    
    def __init__(self, dim=28, hidden=32):
        super().__init__()
        
        # 1. Sensing error 估计（参考 DERNN-LNLT）
        #    以 Phi 为参考做残差学习
        self.delta_phi = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        
        # 2. 退化空间权重（参考 DPU 的 DPB）
        #    输入：cat(Phi, Phi_star) → 2*dim channels
        self.deg_weight = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        
        # 3. 噪声水平估计
        #    从当前特征 f 估计全局噪声水平
        self.sigma_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # σ > 0
        )
    
    def forward(self, f, Phi, Phi_star):
        """
        f:        [B, C, H, W] 当前迭代估计
        Phi:      [B, C, H, W] spatial mask
        Phi_star: [B, C, H, W] 退化 mask（construct_degraded_mask 的输出）
        
        Returns:
            delta_Phi:  [B, C, H, W] sensing error
            deg_weight: [B, C, H, W] 退化权重（0~1）
            sigma:      [B, 1, 1, 1] 噪声水平
        """
        delta_Phi = self.delta_phi(Phi)
        deg_weight = self.deg_weight(torch.cat([Phi, Phi_star], dim=1))
        sigma = self.sigma_est(f).view(-1, 1, 1, 1)
        
        return delta_Phi, deg_weight, sigma
```

### 4.2 三个输出的使用位置

|输出          |用在                   |方式                                        |
|------------|---------------------|------------------------------------------|
|`delta_Phi` |GD step（unfolding.py）|`Phi_eff = Phi + delta_Phi` 替代原始 Phi      |
|`deg_weight`|WPO 输入前（unfolding.py）|`z_clean = z * deg_weight + z`（净化+残差）     |
|`sigma`     |WPO 内部（wpo3d.py）     |`alpha_eff = alpha + lambda_sigma * sigma`|

-----

## 5. 新增文件：model/refinement.py

```python
"""
refinement.py — 轻量局部精化模块

WPO 做全局传播后，补充局部纹理细节。
DWConv 3×3 + GELU + Conv 1×1，约 4.7K 参数（dim=28）。
"""

import torch.nn as nn


class LocalRefinement(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        hidden = dim * expand
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )
    
    def forward(self, x):
        return self.net(x)
```

-----

## 6. 修改文件：model/wpo3d.py

### 6.1 修改概览

从 `version1/stage2/wpo3d.py` 复制，做以下修改：

1. **删除** `mask_mode='B'`（MaskSourceB）相关代码
1. **新增** FBGW 频带引导加权（三选项）
1. **新增** Swin 窗口 WPO（可选）
1. **新增** `sigma` 参数接口（噪声感知阻尼）
1. **删除** `ML_WPO_Block` 和 `WaveMST_ML`（ML 堆砌已否决）
1. **修改** import 路径（`from mask_ops` → `from model.mask_ops`）

### 6.2 WPO3D 的关键修改

**修改 1：forward 签名新增 sigma 参数**

```python
def forward(self, x, mask_spatial, sigma=None):
    B, C, H, W = x.shape
    alpha, vs, vl, t = self._get_effective_params()
    
    # 噪声感知阻尼：sigma 越大 → 阻尼越大 → 传播越保守
    if sigma is not None:
        lambda_sigma = F.softplus(self._lambda_sigma)  # 可学习系数
        alpha = alpha + lambda_sigma * sigma            # [B,1,1,1] 广播
```

对应新增一个可学习参数：

```python
def __init__(self, dim, mask_mode='A', eps=0.1):
    ...
    self._lambda_sigma = nn.Parameter(torch.tensor(0.1))  # 噪声-阻尼耦合系数
```

**修改 2：FBGW 频带引导加权**

在 `_wave_modulate` 方法返回 `out_fft` 之后、`irfftn` 之前，加入 FBGW：

```python
def _apply_fbgw(self, out_fft, u0_fft, sigma, mode='none'):
    """频带引导加权，在 WPO 频域调制之后应用"""
    if mode == 'none':
        return out_fft
    
    if mode == 'snr_adaptive':
        # 方案 A：基于信噪比（零参数）
        # W(ω) = sigmoid((|û₀|² - σ²) / (|û₀|² + σ² + ε))
        power = (u0_fft.abs() ** 2)
        sigma_sq = sigma.view(-1, 1, 1, 1) ** 2 if sigma is not None else 0.01
        W = torch.sigmoid((power - sigma_sq) / (power + sigma_sq + 1e-6))
        return out_fft * W
    
    elif mode == 'learnable_band':
        # 方案 B：可学习频带权重（K 个参数）
        # 按 |ω| 分成 K 个频带，每个频带一个可学习权重
        return out_fft * self._band_weights_expanded
```

方案 B 需要在 `__init__` 中初始化：

```python
if fbgw_mode == 'learnable_band':
    self.num_bands_fbgw = 8
    self._band_weights = nn.Parameter(torch.ones(self.num_bands_fbgw))
```

以及一个预计算频带索引的方法（根据 $|\boldsymbol{\omega}|$ 分 bin）。

**修改 3：Swin 窗口 WPO**

在 `forward` 中，如果启用 Swin，先把输入切成窗，每个窗内独立做 WPO，再重组：

```python
def forward(self, x, mask_spatial, sigma=None):
    if self.use_swin:
        return self._swin_forward(x, mask_spatial, sigma)
    else:
        return self._global_forward(x, mask_spatial, sigma)

def _swin_forward(self, x, mask_spatial, sigma):
    B, C, H, W = x.shape
    ws = self.swin_window_size  # 64
    
    # shift（每隔一层偏移 ws//2）
    if self.swin_shift:
        x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(2, 3))
        mask_spatial = torch.roll(mask_spatial, shifts=(-ws // 2, -ws // 2), dims=(2, 3))
    
    # 切窗：[B, C, H, W] → [B * nH * nW, C, ws, ws]
    nH, nW = H // ws, W // ws
    x_win = x.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws, ws)
    m_win = mask_spatial.view(B, C, nH, ws, nW, ws).permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, ws, ws)
    
    # 每个窗内做 WPO（调用 _global_forward）
    out_win = self._global_forward(x_win, m_win, sigma)
    
    # 重组：[B * nH * nW, C, ws, ws] → [B, C, H, W]
    out = out_win.view(B, nH, nW, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    
    # 反 shift
    if self.swin_shift:
        out = torch.roll(out, shifts=(ws // 2, ws // 2), dims=(2, 3))
    
    return out
```

`swin_shift` 是一个布尔值，在 `WPO3DBlock` 中交替设置（偶数层 False，奇数层 True）：

```python
class WPO3DBlock(nn.Module):
    def __init__(self, dim, mask_mode='A', use_swin=False, swin_window_size=64, swin_shift=False,
                 fbgw_mode='none'):
        ...
        self.wpo = WPO3D(dim, mask_mode=mask_mode,
                         use_swin=use_swin, swin_window_size=swin_window_size,
                         swin_shift=swin_shift, fbgw_mode=fbgw_mode)
```

在 `WaveMST_3D` 构建 blocks 时，交替设置 shift：

```python
blocks = nn.ModuleList([
    WPO3DBlock(dim_stage, mask_mode,
               use_swin=use_swin_wpo,
               swin_window_size=swin_window_size,
               swin_shift=(j % 2 == 1),  # 奇数层 shift
               fbgw_mode=fbgw_mode)
    for j in range(num_blocks[i])
])
```

### 6.3 WaveMST_3D 的修改

新增参数传递：

```python
class WaveMST_3D(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2],
                 mask_mode='A', use_kg=False,
                 use_swin_wpo=False, swin_window_size=64,
                 fbgw_mode='none'):
```

forward 签名新增 sigma：

```python
def forward(self, x, input_mask, sigma=None):
    ...
    for blk in blocks:
        fea = blk(fea, mask_spatial, sigma=sigma)
    ...
```

### 6.4 删除的内容

- `MaskSourceB` 相关的所有 `mask_mode == 'B'` 分支
- `ML_WPO_Block` 类
- `WaveMST_ML` 类
- `enhancement_ops` 的 `use_dispersive` 相关代码

-----

## 7. 修改文件：model/unfolding.py

### 7.1 修改概览

从 `version1/stage2/wpo3d_unfold.py` 重写，核心改动：

1. **GAP → A-HQS**：加入 Nesterov 动量
1. **集成退化估计**：DegradationEstimation 在每个 stage 调用
1. **集成局部精化**：LocalRefinement 在 WPO 之后
1. **sensing error 修正**：GD step 用 `Phi + delta_Phi`
1. **删除** 源项注入、色散修正、ML_Unfold

### 7.2 完整 unfolding 循环

```python
"""
unfolding.py — A-HQS deep unfolding wrapper

每个 stage:
  1. 退化估计 → ΔΦ, deg_weight, σ
  2. Nesterov 动量外推
  3. 修正 GD step（Φ_eff = Φ + ΔΦ）
  4. 初始场净化（deg_weight 加权）
  5. WPO 传播（σ 控制阻尼）
  6. 局部精化（DWConv FFN）
"""

import torch
import torch.nn as nn
from model.wpo3d import WaveMST_3D, WaveMST_KG
from model.degradation import DegradationEstimation, construct_degraded_mask
from model.refinement import LocalRefinement
from model.utils import (
    shift_batch, shift_back_batch,
    mul_Phi_f, mul_PhiT_residual,
    ParaEstimator,
)


class WPO_Unfold(nn.Module):
    """A-HQS Unfolding：退化估计 → 动量 → GD → 净化 → WPO → 精化
    
    Args:
        dim, unet_stage, num_blocks: 传给 WaveMST_3D
        use_kg: True → KG 方程
        num_stages: K（unfolding stage 数）
        share_weights: 共享 prior + degradation + refinement
        use_swin_wpo, swin_window_size: Swin-WPO 选项
        fbgw_mode: FBGW 选项
        size: crop_size（用于 shift_back）
    """
    
    def __init__(self, dim=28, unet_stage=3, num_blocks=None,
                 use_kg=False,
                 num_stages=5, share_weights=True,
                 use_swin_wpo=False, swin_window_size=64,
                 fbgw_mode='none',
                 size=256, len_shift=2):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.num_stages = num_stages
        self.share_weights = share_weights
        self.nC = dim
        self.size = size
        self.len_shift = len_shift
        
        mask_mode = 'D' if use_kg else 'A'
        
        # ── ParaEstimator：每 stage 独立（即使 share_weights） ──
        self.rho_estimators = nn.ModuleList([
            ParaEstimator(in_nc=dim) for _ in range(num_stages)
        ])
        
        # ── Nesterov 动量系数：每 stage 一个可学习标量 ──
        self.betas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(num_stages)
        ])
        
        # ── 退化估计 ──
        if share_weights:
            self.deg_est = DegradationEstimation(dim)
        else:
            self.deg_ests = nn.ModuleList([
                DegradationEstimation(dim) for _ in range(num_stages)
            ])
        
        # ── Prior: WaveMST_3D / KG ──
        prior_class = WaveMST_KG if use_kg else WaveMST_3D
        prior_kwargs = dict(
            dim=dim, stage=unet_stage, num_blocks=num_blocks,
            mask_mode=mask_mode,
            use_swin_wpo=use_swin_wpo,
            swin_window_size=swin_window_size,
            fbgw_mode=fbgw_mode,
        )
        if share_weights:
            self.shared_prior = prior_class(**prior_kwargs)
            self.priors = None
        else:
            self.priors = nn.ModuleList([
                prior_class(**prior_kwargs) for _ in range(num_stages)
            ])
            self.shared_prior = None
        
        # ── 局部精化 ──
        if share_weights:
            self.local_refine = LocalRefinement(dim)
        else:
            self.local_refines = nn.ModuleList([
                LocalRefinement(dim) for _ in range(num_stages)
            ])
        
        # ── 初始化卷积 ──
        self.initial_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)
    
    def _get_module(self, name, k):
        """获取第 k 个 stage 的模块（shared 或独立）"""
        if self.share_weights:
            return getattr(self, name)
        else:
            return getattr(self, name + 's')[k]
    
    def forward(self, g, input_mask):
        """
        g:          [B, 1, H, W'] measurement
        input_mask: (Phi, PhiPhiT)
            Phi:    [B, C, H, W] spatial mask
            PhiPhiT:[B, 1, H, W'] 预计算
        
        Returns: list of [B, C, H, W]，每个 stage 的输出
        """
        Phi, PhiPhiT = input_mask
        Phi_shift = shift_batch(Phi, self.len_shift)
        
        # 预计算退化 mask（只算一次）
        Phi_star = construct_degraded_mask(Phi, self.len_shift)
        
        # 初始化
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)
        f0 = shift_back_batch(temp_g, self.len_shift, self.size)
        f = self.initial_conv(torch.cat([f0, Phi], dim=1))
        
        f_prev = f.clone()  # 动量用
        outputs = []
        
        for k in range(self.num_stages):
            # ── 1. 退化估计 ──
            deg_est = self._get_module('deg_est', k)
            delta_Phi, deg_weight, sigma = deg_est(f, Phi, Phi_star)
            
            # ── 2. Nesterov 动量外推 ──
            beta_k = torch.sigmoid(self.betas[k])  # 限制在 (0, 1)
            f_momentum = f + beta_k * (f - f_prev)
            f_prev = f.clone()
            
            # ── 3. 修正 GD step ──
            rho_k = self.rho_estimators[k](f_momentum)
            
            Phi_eff = Phi + delta_Phi  # sensing error 修正
            Phi_eff_shift = shift_batch(Phi_eff, self.len_shift)
            
            Phi_f = mul_Phi_f(Phi_eff_shift, f_momentum, self.len_shift)
            residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
            residual = residual.clamp(min=-10, max=10)
            z = f_momentum + rho_k * mul_PhiT_residual(
                Phi_eff_shift, residual, self.len_shift, self.size
            )
            
            # ── 4. 初始场净化 ──
            z_clean = z * deg_weight + z  # 退化加权 + 残差
            
            # ── 5. WPO 传播 ──
            prior = self._get_module('shared_prior' if self.share_weights else 'prior', k)
            if self.share_weights:
                prior = self.shared_prior
            else:
                prior = self.priors[k]
            f_wave = prior(z_clean, Phi, sigma=sigma)
            
            # ── 6. 局部精化 ──
            refine = self._get_module('local_refine', k)
            f_local = refine(f_wave)
            
            # ── 7. 输出（三路残差） ──
            f = z + f_wave + f_local
            
            outputs.append(f)
        
        return outputs
```

-----

## 8. 修改文件：train.py

### 8.1 核心结构

```python
"""
train.py — 训练入口
读取 config.yaml 和 __init__.py 的配置
"""

import yaml
import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path

# 读取配置
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

from __init__ import *  # 模型开关

os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']


def build_model():
    if USE_UNFOLDING:
        from model.unfolding import WPO_Unfold
        return WPO_Unfold(
            dim=cfg['dim'],
            unet_stage=cfg['unet_stage'],
            num_blocks=cfg['num_blocks'],
            use_kg=USE_KG,
            num_stages=NUM_STAGES,
            share_weights=SHARE_STAGE_WEIGHTS,
            use_swin_wpo=USE_SWIN_WPO,
            swin_window_size=SWIN_WINDOW_SIZE,
            fbgw_mode=WPO_FBGW_MODE,
            size=cfg['crop_size'],
        )
    else:
        from model.wpo3d import WaveMST_3D, WaveMST_KG
        cls = WaveMST_KG if USE_KG else WaveMST_3D
        return cls(
            dim=cfg['dim'],
            stage=cfg['unet_stage'],
            num_blocks=cfg['num_blocks'],
            use_swin_wpo=USE_SWIN_WPO,
            swin_window_size=SWIN_WINDOW_SIZE,
            fbgw_mode=WPO_FBGW_MODE,
        )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def print_config(model):
    """打印当前组合"""
    print("=" * 60)
    print(f"当前配置组合:")
    print(f"  KG方程:     {'是' if USE_KG else '否'}")
    print(f"  FBGW:       {WPO_FBGW_MODE}")
    print(f"  Swin-WPO:   {'是 (ws={SWIN_WINDOW_SIZE})' if USE_SWIN_WPO else '否'}")
    if USE_UNFOLDING:
        print(f"  展开:       {NUM_STAGES} stage, "
              f"{'共享' if SHARE_STAGE_WEIGHTS else '独立'}权重, A-HQS+动量")
        print(f"  多阶段损失: {'是' if MULTI_STAGE_LOSS else '否'}")
    else:
        print(f"  展开:       无 (端到端)")
    print(f"  参数量:     {count_params(model):.2f}M")
    print("=" * 60)


# ── 损失函数 ──

def rmse_loss(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt))

def multi_stage_loss(outputs, gt):
    K = len(outputs)
    loss = rmse_loss(outputs[-1], gt)
    if K >= 2: loss = loss + 0.7 * rmse_loss(outputs[-2], gt)
    if K >= 3: loss = loss + 0.5 * rmse_loss(outputs[-3], gt)
    if K >= 4: loss = loss + 0.3 * rmse_loss(outputs[-4], gt)
    return loss


# ── 训练循环 ──
# （从 version1/stage2/train.py 的训练循环复制，修改以下部分：
#    1. 读 config.yaml 的路径
#    2. 用 build_model() 创建模型
#    3. print_config() 打印组合
#    4. 调用逻辑与 stage2 一致：unfolding 返回 list，e2e 返回 tensor）
```

### 8.2 训练循环中的 unfolding 路径

和 stage2 的逻辑完全一致——unfolding 模型返回 list，用 multi_stage_loss：

```python
if USE_UNFOLDING:
    outputs = model(g, (Phi, PhiPhiT))
    if MULTI_STAGE_LOSS:
        loss = multi_stage_loss(outputs, gt)
    else:
        loss = rmse_loss(outputs[-1], gt)
    pred = outputs[-1]
else:
    pred = model(input_meas, shift_mask)
    loss = rmse_loss(pred, gt)
```

-----

## 9. 修改文件：test.py

和 train.py 共享 `__init__.py` 的配置。从 `BEST_CKPT` 加载 checkpoint。

```python
from __init__ import *

model = build_model().cuda()
model.load_state_dict(torch.load(BEST_CKPT))
model.eval()
```

-----

## 10. 修改文件：dataset.py

从 `version1/stage2/dataset.py` 复制，修改数据路径从 `config.yaml` 读取：

```python
import yaml
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = Path(cfg['data_root'])
TRAIN_PATH = Path(cfg['train_path'])
# ... 其余不变
```

-----

## 11. 关键陷阱

### 11.1 Swin-WPO 窗口大小与图像大小的整除

64×64 窗，256×256 图像 → 4×4 个窗，可整除。

但 U-Net 下采样后：128×128 → 2×2 个窗，可整除。64×64 → 1×1 个窗，等于全局 WPO，Swin 退化为普通 WPO。

**所以 Swin-WPO 只在 U-Net 的前两层（256×256 和 128×128）有效。** bottleneck（64×64）自动退化为全局——这是正确的行为（深层应该全局建模）。

### 11.2 A-HQS 的 Phi_eff 用于 GD step

`Phi_eff = Phi + delta_Phi` 需要重新计算 `Phi_eff_shift = shift_batch(Phi_eff)`。**不能复用预计算的 Phi_shift**——因为 delta_Phi 每个 stage 可能不同（非共享时）。

如果 share_weights=True，delta_Phi 每个 stage 相同，**可以预计算一次** Phi_eff_shift 复用。但为简洁起见，先不优化。

### 11.3 动量的初始化

`self.betas` 初始化为 0（`torch.tensor(0.0)`），经 sigmoid 后初始值 = 0.5。

训练初期动量可能不稳定——如果出现发散，改为初始化 `torch.tensor(-2.0)`（sigmoid 后 ≈ 0.12，接近零动量）。

### 11.4 退化 mask 的预计算

`construct_degraded_mask(Phi)` 只依赖 Phi 和 len_shift，不依赖当前估计 $f$。**可以在 forward 开头算一次**，所有 stage 共享。代码中已经这样做了。

### 11.5 sigma 的广播

`sigma` shape 是 `[B, 1, 1, 1]`，传入 WPO3D 后需要和 `alpha`（标量）相加。PyTorch 自动广播，但要注意 `alpha` 是通过 softplus 得到的标量——加法后变成 `[B, 1, 1, 1]`，后续的频域调制需要确保 shape 兼容。

在 `_wave_modulate` 中，`alpha` 参与计算 `eta = omega_sq - (alpha/2)**2`。如果 `alpha` 变成 `[B,1,1,1]` 而 `omega_sq` 是 `[C,H,W']`，需要对 `alpha` unsqueeze 到 `[B,1,1,1,1]` 或其他兼容 shape。

**最安全的做法**：在 forward 中把 sigma 影响的 alpha_eff 传入 `_wave_modulate`，确保 shape 一致：

```python
alpha_eff = alpha + lambda_sigma * sigma.mean()  # 取 batch 均值，保持标量
```

-----

## 12. 开发顺序与验证

### 12.1 Phase 1（2天）：基础搭建

- [ ] 创建 version2/ 目录结构
- [ ] 复制文件，修改 import 路径
- [ ] 写 config.yaml 和 `__init__.py`
- [ ] 确认纯 WPO（无新模块）能跑通：`USE_UNFOLDING=True`, 其余 `none/False`
- [ ] **验收**：跑 5 epoch，PSNR ≈ stage2 同 epoch（确认复制无误）

### 12.2 Phase 2（2天）：逐个加模块

**每加一个模块，跑 10 epoch 验证是否有效**：

- [ ] 加 DegradationEstimation → 跑 10ep → 看 PSNR 是否 > 纯 WPO 的 10ep
- [ ] 加 LocalRefinement → 跑 10ep → 看增量
- [ ] 加 Nesterov 动量 → 跑 10ep → 看增量
- [ ] 加 FBGW snr_adaptive → 跑 10ep → 看增量
- [ ] 加 Swin-WPO → 跑 10ep → 看增量

**如果某个模块 10ep 后无增量或负增量 → 删除，不强行保留。**

### 12.3 Phase 3（1周）：最佳组合跑满

选出有效模块的组合，跑 300 epoch：

- [ ] 最佳组合 5-stage shared → 300ep → 预期 > 38.5 dB
- [ ] 最佳组合 5-stage non-shared → 300ep → 预期 > 39.0 dB
- [ ] KG 版本 → 300ep → 关注 SAM

### 12.4 验收标准

|配置                           |下限              |说明                 |
|-----------------------------|----------------|-------------------|
|纯 WPO 5stg (baseline)        |38.21           |已有结果               |
|+ 退化估计 @10ep                 |> baseline @10ep|退化估计有效             |
|+ 全部模块 5stg shared @300ep    |**> 38.8**      |超过 baseline 0.6+ dB|
|+ 全部模块 5stg non-shared @300ep|**> 39.2**      |逼近 DPU 水平          |