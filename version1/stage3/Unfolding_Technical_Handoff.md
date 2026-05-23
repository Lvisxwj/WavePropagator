# 3D-WPO/KG Unfolding 升级阶段一 — Claude Code Handoff

> **目的**：在已有 WaveMST 代码基础上，把 3D-WPO Pure 和 3D-WPO-KG 这两个物理可解释模型升级为 **deep unfolding** 框架，预期 PSNR 从 34.7 dB 提升到 38+ dB。
> **要求**：从G:\MachineLearning\CASSI中复制走需要的代码到G:\MachineLearning\CASSI\stage2后再修改，这次的参考代码在\stage2\DPU和\stage2\SSR。reslut等东西也放在stage2这个目录，数据集不动，所以path一样，先查看path的逻辑，看是软编码还是怎么的，修改好
>
> **已有基础**：`wpo3d.py`（含 `WaveMST_3D` 和 `WaveMST_KG`）已跑通并训练完成。本次只新增 unfolding 包装层，**不修改原有 WPO 模块内部**。
>
> **参考代码**：用户已 clone DPU（`./DPU/`）和 SSR（`./SSR/`）到本地。本文档大量引用这两个仓库的具体实现。
>
> **核心理论依据**：详见 `3D_WPO_KG_Improvement_Analysis.md` 第 3、7、11 节。

---

## 目录

1. [改造目标与架构选择](#1-改造目标与架构选择)
2. [Unfolding 数学框架](#2-unfolding-数学框架)
3. [新增/修改文件清单](#3-新增修改文件清单)
4. [新增文件 1：unfolding_ops.py](#4-新增文件-1unfolding_opspy)
5. [新增文件 2：wpo3d_unfold.py](#5-新增文件-2wpo3d_unfoldpy)
6. [修改文件：dataset.py](#6-修改文件datasetpy)
7. [修改文件：train.py](#7-修改文件trainpy)
8. [修改文件：test.py](#8-修改文件testpy)
9. [训练超参数配置建议](#9-训练超参数配置建议)
10. [关键陷阱与调试要点](#10-关键陷阱与调试要点)
11. [验证清单与开发顺序](#11-验证清单与开发顺序)
12. [对照参考：DPU 与 SSR 的关键代码片段](#12-对照参考dpu-与-ssr-的关键代码片段)

---

## 1. 改造目标与架构选择

### 1.1 目标

把 `WaveMST_3D` / `WaveMST_KG` 从单 forward pass 升级为 K-stage unfolding 网络。每个 stage 包含：

1. **GD step（数据保真）**：测量值 $g$ 的反投影约束
2. **Prior step**：3D-WPO 或 3D-WPO-KG 模块作为可学习先验
3. **可选：Source term**：$\Phi^T g$ 作为波动方程的源项注入

**关键原则**：原有 WPO 模块**完全不动**。只在外部加一层包装。

### 1.2 两种 Unfolding 风格的选择

| 风格 | 代表 | 公式核心 | 复杂度 |
|------|------|---------|-------|
| **GAP（推荐入门）** | SSR | $z = f + \rho \Phi^T(g - \Phi f)/(\Phi\Phi^T)$，$f^{k+1} = \text{Prior}(z)$ | 简单 |
| **ADMM** | DPU | 含 Lagrange multiplier $y$ 和 dual prior $r$ | 复杂 |

**推荐采用 GAP 风格**。理由：
- 实现最简单（约 50 行新代码）
- 数学公式干净，与我们的 prior network 解耦清晰
- SSR 用 GAP 也达到了 SOTA（40.69 dB），说明 GAP 框架本身够用
- DPU 的优势主要来自 Focused Attention 和双 prior，那是后续的事

### 1.3 用户配置接口

train.py 顶部新增配置项：

```python
# Unfolding 配置
USE_UNFOLDING = True       # True 启用 unfolding，False 退回单 stage
NUM_STAGES = 5             # unfolding stage 数：3/5/7/9
SHARE_STAGE_WEIGHTS = False  # True 所有 stage 共享 WPO 权重，False 每 stage 独立
USE_SOURCE_INJECTION = False  # 是否注入 Φ^T g 作为源项（实验性）
MULTI_STAGE_LOSS = True    # 是否用多 stage 加权损失（DPU 风格）
```

`NUM_STAGES` 用户可自由选择。`SHARE_STAGE_WEIGHTS=True` 可在不增加参数量的情况下迭代精化（参数量保持 0.79M），`False` 时参数量 ≈ 0.79M × NUM_STAGES。

---

## 2. Unfolding 数学框架

### 2.1 GD Step（数据保真）

CASSI 的测量模型：$g = \Phi f + n$，其中 $\Phi$ 是 sensing matrix（含 mask + shift + sum）。

GD step 闭式解：
$$z = f + \rho \Phi^T \frac{g - \Phi f}{\Phi \Phi^T}$$

其中：
- $\Phi^T g$ 是测量值反投影到 HSI 空间（"shift_back"）
- $\Phi \Phi^T$ 是预计算的对角矩阵（每个像素位置的 mask 平方和）
- $\rho$ 是可学习步长，由小网络 `Para_Estimator` 从 $f$ 预测

**关键工程操作**（参考 SSR 的 `mul_PhiTg` 和 `mul_Phif`）：

```python
def mul_Phi_f(self, Phi_shift, f):
    """计算 Φf：先 shift 再加权再 sum"""
    f_shift = self.shift(f, len_shift=2)        # [B, C, H, W+54]
    Phi_f = Phi_shift * f_shift                  # 逐元素乘
    Phi_f = torch.sum(Phi_f, dim=1, keepdim=True) # 沿光谱维 sum -> [B, 1, H, W+54]
    return Phi_f

def mul_PhiT_residual(self, Phi_shift, residual):
    """计算 Φ^T (g - Φf)：先广播再 shift_back"""
    temp = residual.repeat(1, self.nC, 1, 1)    # [B, C, H, W+54]
    PhiT = temp * Phi_shift                      # 加权
    PhiT = self.shift_back(PhiT)                # [B, C, H, W]
    return PhiT
```

### 2.2 Prior Step（3D-WPO/KG）

```python
f_new = WPO3D(z, mask)   # 我们已有的 WaveMST_3D 模块
```

注意原 `WaveMST_3D` 接收 `(x, input_mask)`，其中 `input_mask` 是 shifted mask。在 unfolding 中我们传入未 shifted 的 `Phi`（spatial mask），WPO 内部自己处理。

### 2.3 完整 K-stage 流程

```
输入: g [B, 1, H, W+(C-1)*step], Phi [B, C, H, W], PhiPhiT [B, 1, H, W+(C-1)*step]

预处理:
  Phi_shift = shift(Phi)          # [B, C, H, W+(C-1)*step]
  f0 = shift_back(g.repeat(1,C,1,1) / C * 2)  # 初始估计
  f = initial_conv(cat([f0, Phi]))  # [B, C, H, W]

迭代 (k = 0, ..., K-1):
  rho_k = rho_estimator[k](f)     # 学习的步长
  Phi_f = mul_Phi_f(Phi_shift, f)  # [B, 1, H, W+(C-1)*step]
  residual = (g - Phi_f) / PhiPhiT
  z = f + rho_k * mul_PhiT_residual(Phi_shift, residual)  # GD step
  f = WPO3D[k](z, Phi)            # Prior step
  out_list.append(f)

返回: out_list (用于多 stage loss) 或 out_list[-1] (推理)
```

### 2.4 多 stage Loss（重要）

DPU 实证证明，单纯用最后 stage 的 loss 训练 unfolding 不稳定。需要对最后几个 stage 加权：

$$\mathcal{L} = \sqrt{\text{MSE}(f^K, \text{GT})} + 0.7 \sqrt{\text{MSE}(f^{K-1}, \text{GT})} + 0.5 \sqrt{\text{MSE}(f^{K-2}, \text{GT})} + 0.3 \sqrt{\text{MSE}(f^{K-3}, \text{GT})}$$

只有 K≥4 时启用。K<4 时退化为单 stage loss。

---

## 3. 新增/修改文件清单

```
新增：
  unfolding_ops.py     ← shift/shift_back/mul_Phi_f/mul_PhiT_residual + ParaEstimator
  wpo3d_unfold.py      ← WaveMST_3D_Unfold（包装类）

修改：
  dataset.py           ← 增加 PhiPhiT 输出（仿造 DPU 的 Phi_s_batch）
  train.py             ← 增加 unfolding 配置 + multi-stage loss + Model 索引扩展
  test.py              ← 增加 unfolding 模型加载与推理

不变：
  wpo3d.py             ← 完全不动（这是关键，物理算子不修改）
  mask_ops.py, loss.py, viz.py, mst.py, wpo_smsa.py, wpo_mamba.py, helmholtz_ops.py 等
```

---

## 4. 新增文件 1：unfolding_ops.py

### 4.1 功能与对应关系

包含 unfolding 框架所需的所有"非 prior"组件：

- `shift` / `shift_back`：CASSI 的 dispersion 模拟与逆操作（与 dataset.py 中的 shift_3 等价）
- `mul_Phi_f`：计算 $\Phi f$
- `mul_PhiT_residual`：计算 $\Phi^T (g - \Phi f)$
- `ParaEstimator`：从当前 $f$ 预测步长 $\rho_k$（参考 SSR 的 `Para_Estimator`）

### 4.2 ParaEstimator 实现

参考 SSR `Model.py` 第 65-87 行。设计一个轻量 CNN：

```python
class ParaEstimator(nn.Module):
    """从当前迭代值 f 预测步长 rho_k
    
    输出 [B, 1, 1, 1] 标量，作用于 GD step 的修正幅度
    """
    def __init__(self, in_nc=28, channel=32):
        super().__init__()
        self.fusion = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Sequential(
            nn.Conv2d(channel, channel*2, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel*2, channel*4, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(channel*4, 1, 1, 1, 0, bias=True)
    
    def forward(self, f):
        # f: [B, 28, H, W]
        x = self.fusion(f)
        x = self.down_sample(x)
        x = self.avg_pool(x)
        rho = self.out(x)   # [B, 1, 1, 1]
        return torch.sigmoid(rho)  # 限制在 (0, 1)
```

注意：DPU 用的是 `Mu_Estimator`（输出 mu，penalty 参数），SSR 用 `Para_Estimator`（输出 rho，步长）。两者本质相同，名字不同。我们用 SSR 的命名。

### 4.3 Shift 工具函数

直接从 DPU `Utils.py` 移植，注意改成 batched 版本（DPU 的 `shift_3` 是单样本，`shift_4` 是 batch）：

```python
def shift_batch(f, len_shift=2, num_bands=28):
    """Batched shift: [B, C, H, W] -> [B, C, H, W + (C-1)*len_shift]
    
    每个波段 c 沿宽度方向右移 c * len_shift 像素
    参考 DPU/Utils.py shift_4
    """
    B, C, H, W = f.shape
    pad_w = (C - 1) * len_shift
    shifted = torch.zeros(B, C, H, W + pad_w, device=f.device, dtype=f.dtype)
    for c in range(C):
        shifted[:, c, :, c*len_shift:c*len_shift+W] = f[:, c, :, :]
    return shifted

def shift_back_batch(f, len_shift=2, num_bands=28, output_w=256):
    """Batched shift_back: [B, C, H, W'] -> [B, C, H, W]
    
    参考 DPU/Utils.py shift_back
    """
    f = f.clone()  # 避免修改输入
    for c in range(num_bands):
        f[:, c, :, :] = torch.roll(f[:, c, :, :], shifts=-len_shift*c, dims=2)
    return f[:, :, :, :output_w]
```

**注意**：torch.roll 是循环移位，但只要后续切片到 `output_w`，多余部分就丢弃了，等价于普通移位。

### 4.4 Phi 操作

```python
def mul_Phi_f(Phi_shift, f, len_shift=2, num_bands=28):
    """计算 Φf
    Phi_shift: [B, C, H, W'] 已 shifted 的 mask
    f: [B, C, H, W] 当前估计
    返回: [B, 1, H, W']
    """
    f_shift = shift_batch(f, len_shift, num_bands)  # [B, C, H, W']
    Phi_f = Phi_shift * f_shift
    Phi_f = torch.sum(Phi_f, dim=1, keepdim=True)
    return Phi_f

def mul_PhiT_residual(Phi_shift, residual, len_shift=2, num_bands=28, output_w=256):
    """计算 Φ^T r，其中 r = (g - Φf) / (Φ Φ^T)
    Phi_shift: [B, C, H, W']
    residual: [B, 1, H, W']
    返回: [B, C, H, W]
    """
    temp = residual.repeat(1, num_bands, 1, 1)
    PhiT = temp * Phi_shift
    PhiT = shift_back_batch(PhiT, len_shift, num_bands, output_w)
    return PhiT
```

### 4.5 完整文件结构

```python
# unfolding_ops.py
import torch
import torch.nn as nn

def shift_batch(f, len_shift=2, num_bands=28):
    ...

def shift_back_batch(f, len_shift=2, num_bands=28, output_w=256):
    ...

def mul_Phi_f(Phi_shift, f, len_shift=2, num_bands=28):
    ...

def mul_PhiT_residual(Phi_shift, residual, len_shift=2, num_bands=28, output_w=256):
    ...

class ParaEstimator(nn.Module):
    ...
```

---

## 5. 新增文件 2：wpo3d_unfold.py

### 5.1 核心类设计

```python
import torch
import torch.nn as nn
from wpo3d import WaveMST_3D, WaveMST_KG  # 复用已有的 prior network
from unfolding_ops import (
    shift_batch, shift_back_batch, 
    mul_Phi_f, mul_PhiT_residual, 
    ParaEstimator
)


class WaveMST_3D_Unfold(nn.Module):
    """3D-WPO 的 K-stage unfolding 包装类
    
    架构: K 个 stage,每个 stage = GD step + WPO3D prior
    
    Args:
        num_stages: K, unfolding 的 stage 数
        share_weights: 是否所有 stage 共享 WPO 权重
        use_kg: True 用 KG 方程,False 用普通 WPO
        其他参数透传给 WaveMST_3D
    """
    def __init__(self, dim=28, stage=2, num_blocks=[2,2,2],
                 num_stages=5, share_weights=False, use_kg=False,
                 mask_mode='A', size=256, len_shift=2):
        super().__init__()
        self.num_stages = num_stages
        self.share_weights = share_weights
        self.nC = dim
        self.size = size
        self.len_shift = len_shift
        
        # ParaEstimator: 每个 stage 一个
        self.rho_estimators = nn.ModuleList([
            ParaEstimator(in_nc=dim) for _ in range(num_stages)
        ])
        
        # Prior networks: WPO3D
        prior_class = WaveMST_KG if use_kg else WaveMST_3D
        if share_weights:
            # 所有 stage 共享同一个 WPO
            self.shared_prior = prior_class(
                dim=dim, stage=stage, num_blocks=num_blocks, mask_mode=mask_mode
            )
            self.priors = None
        else:
            # 每个 stage 独立 WPO
            self.priors = nn.ModuleList([
                prior_class(dim=dim, stage=stage, num_blocks=num_blocks, mask_mode=mask_mode)
                for _ in range(num_stages)
            ])
            self.shared_prior = None
        
        # 初始化卷积:把 [shift_back(g), Phi] 融合到 [B, C, H, W]
        self.initial_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)
    
    def get_prior(self, k):
        """获取第 k 个 stage 的 prior network"""
        return self.shared_prior if self.share_weights else self.priors[k]
    
    def forward(self, g, input_mask):
        """
        Args:
            g: [B, 1, H, W'] 测量值 (W' = W + (C-1)*len_shift)
            input_mask: tuple (Phi, PhiPhiT)
                Phi: [B, C, H, W] spatial mask
                PhiPhiT: [B, 1, H, W'] 预计算的 Φ Φ^T
        
        Returns:
            list of [B, C, H, W]: 每个 stage 的输出 (length = num_stages)
            训练时全部用于 multi-stage loss; 推理时只取最后一个
        """
        Phi, PhiPhiT = input_mask
        Phi_shift = shift_batch(Phi, self.len_shift, self.nC)
        
        # 初始化 f0
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)
        f0 = shift_back_batch(temp_g, self.len_shift, self.nC, self.size)
        f = self.initial_conv(torch.cat([f0, Phi], dim=1))
        
        outputs = []
        for k in range(self.num_stages):
            # GD step
            rho_k = self.rho_estimators[k](f)  # [B, 1, 1, 1]
            Phi_f = mul_Phi_f(Phi_shift, f, self.len_shift, self.nC)
            residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
            z = f + rho_k * mul_PhiT_residual(
                Phi_shift, residual, self.len_shift, self.nC, self.size
            )
            
            # Prior step (3D-WPO)
            # 注意: WaveMST_3D.forward 签名是 (x, input_mask)
            # input_mask 在原代码中是 shifted mask, 但我们传 spatial mask
            # 看 wpo3d.py 内部如何处理 mask, 确保兼容
            f = self.get_prior(k)(z, Phi)  # 假设 prior 接收 spatial mask
            
            outputs.append(f)
        
        return outputs


class WaveMST_KG_Unfold(WaveMST_3D_Unfold):
    """KG 方程的 unfolding 版本——和 WaveMST_3D_Unfold 唯一区别是 use_kg=True"""
    def __init__(self, **kwargs):
        kwargs['use_kg'] = True
        super().__init__(**kwargs)
```

### 5.2 关键设计决策

**(1) 为什么 prior network 接收 spatial mask 而非 shifted mask？**

原 `WaveMST_3D.forward(x, input_mask)` 中，`input_mask` 用法见 `wpo3d.py`。如果原代码内部做了 shift_back 转 spatial mask，则我们直接传 Phi（spatial）即可。如果原代码期望 shifted mask，则需要在 unfold 中先做 shift。

**Claude Code 任务**：在实现时先 `view wpo3d.py`，确认 `WaveMST_3D.forward` 中 `input_mask` 是如何使用的。如果它内部做了 `shift_back(input_mask)`，那么需要传 shifted mask；否则传 spatial mask。**两种方案任选其一即可，但要保持一致。**

**(2) 为什么 share_weights 默认 False？**

DPU 和 SSR 都用每 stage 独立权重——参数量增加换更强表达力。但参数量从 0.79M 到 0.79M × 5 = 4M，超过原 KG 5 倍。`share_weights=True` 时所有 stage 共享，参数量不变，仅靠迭代次数提升性能（DPU 表 1 显示 RDLUF 5stg 共享版本仍优于其他方法）。

让用户在 train.py 里自行选择。

**(3) 为什么用 sigmoid 限制 rho？**

GD step 的步长太大会让 z 偏离当前估计太远，破坏 prior 已学到的信息；太小则收敛慢。sigmoid 限制在 (0,1) 是稳妥设置。

---

## 6. 修改文件：dataset.py

### 6.1 必须的修改

原 `dataset.py` 返回 `(input_meas, gt, mask3d, shift_mask)` 四个张量（具体名字以你的实现为准）。unfolding 需要额外的 `PhiPhiT` 张量。

新增计算：

```python
def compute_PhiPhiT(mask3d, len_shift=2):
    """计算 Φ Φ^T 用于 GD step 的分母
    
    Φ Φ^T 是个对角矩阵, 对角元 = 每个像素位置上 mask^2 在所有波段的和
    在 shifted 空间中:
        PhiPhiT[h, w] = sum_c mask3d[c, h, w - c*len_shift]^2
    
    输出 shape: [1, H, W'] 与 g 同 shape (broadcast over batch)
    """
    nC, H, W = mask3d.shape
    Phi_shifted_sq = torch.zeros(nC, H, W + (nC-1)*len_shift, device=mask3d.device)
    for c in range(nC):
        Phi_shifted_sq[c, :, c*len_shift:c*len_shift+W] = mask3d[c, :, :] ** 2
    PhiPhiT = torch.sum(Phi_shifted_sq, dim=0, keepdim=True)  # [1, H, W']
    PhiPhiT[PhiPhiT == 0] = 1.0  # 防止除零(对应 mask 全零的位置)
    return PhiPhiT
```

参考 DPU `Dataset.py` 第 109-110 行：
```
Phi_s_batch = torch.sum(shift_3(Phi_batch, 2) ** 2, 0)
Phi_s_batch[Phi_s_batch == 0] = 1
```

### 6.2 数据加载流程修改

如果你的 dataset.py 是这样：

```python
def __getitem__(self, index):
    # ... 加载 GT, mask, augmentation
    input_meas = gen_meas(gt, mask3d, ...)
    return input_meas, gt, mask3d, shift_mask
```

改为：

```python
def __getitem__(self, index):
    # ... 加载 GT, mask, augmentation
    g = gen_measurement(gt, mask3d, len_shift=2)  # [1, H, W']  (shifted, summed)
    PhiPhiT = compute_PhiPhiT(mask3d, len_shift=2)  # [1, H, W']
    return g, gt, mask3d, PhiPhiT
```

**关键变化**：原本输入是 `input_meas`（shape `[C, H, W]`，即 reverse 后的初始化），现在直接给 unfolding 模型**未经过 reverse 的 g**（shape `[1, H, W']`）。这是因为 unfolding 内部第一步会自己做 reverse 和 initial_conv。

### 6.3 兼容性策略

为了不破坏已训练的 4 个 Wave 模型，**保留两套数据接口**：

```python
def __getitem__(self, index):
    # ... 通用部分: 加载 GT, mask
    
    if self.return_unfolding_data:
        # Unfolding 模型: 返回原始 g 和 PhiPhiT
        g = gen_measurement(gt, mask3d, len_shift=2)  
        PhiPhiT = compute_PhiPhiT(mask3d, len_shift=2)
        return g, gt, mask3d, PhiPhiT
    else:
        # End-to-end 模型 (Model 0-6): 返回 reverse 后的 input_meas
        input_meas = gen_meas_reversed(gt, mask3d, ...)
        return input_meas, gt, mask3d, shift_mask
```

train.py 根据模型类型设置 `dataset.return_unfolding_data = True/False`。

---

## 7. 修改文件：train.py

### 7.1 配置区扩展

```python
# ============ CONFIG ============
MODEL_INDEX = 7    # 0-6 是已有模型, 7+ 是 unfolding
GPU_ID = '0'
BATCH_SIZE = 5
MAX_EPOCH = 300
LEARNING_RATE = 4e-4
# ... 已有配置 ...

# Unfolding 专用配置 (只对 MODEL_INDEX >= 7 生效)
NUM_STAGES = 5              # unfolding stage 数: 3/5/7/9
SHARE_STAGE_WEIGHTS = False # 是否共享 stage 权重
USE_SOURCE_INJECTION = False  # 实验性: 在 prior network 中注入 Φ^T g 作为源项
MULTI_STAGE_LOSS = True     # 多 stage 加权损失
# ================================

MODELS = {
    0: ('WaveMST_3D',           '3d_wpo_pure'),
    1: ('WaveMST_KG',           '3d_wpo_kg'),
    2: ('WaveMST_Parallel',     '3d_wpo_smsa'),
    3: ('WaveMST_Mamba',        '2d_wpo_mamba'),
    4: ('WaveMST_Phys',         'h2_alpha_phys'),
    5: ('Helmholtzformer',      'h1_gamma_helm_pure'),
    6: ('WaveMST_Helm',         'h2_gamma_main'),
    # 新增:
    7: ('WaveMST_3D_Unfold',    '3d_wpo_unfold'),       # Unfolding 版 3D-WPO
    8: ('WaveMST_KG_Unfold',    '3d_wpo_kg_unfold'),    # Unfolding 版 KG (主推)
}
```

### 7.2 build_model 扩展

```python
def build_model(index):
    if index in [0, 1, 2, 3]:
        # 已有 Wave 模型 (代码不变)
        ...
    elif index in [4, 5, 6]:
        # 已有 Helmholtz 模型 (代码不变)
        ...
    
    # 新增: unfolding 模型
    elif index == 7:
        from wpo3d_unfold import WaveMST_3D_Unfold
        return WaveMST_3D_Unfold(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            num_stages=NUM_STAGES,
            share_weights=SHARE_STAGE_WEIGHTS,
            use_kg=False,
            mask_mode=MASK_MODE,
            size=CROP_SIZE,
            len_shift=2,
        )
    elif index == 8:
        from wpo3d_unfold import WaveMST_KG_Unfold
        return WaveMST_KG_Unfold(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            num_stages=NUM_STAGES,
            share_weights=SHARE_STAGE_WEIGHTS,
            mask_mode=MASK_MODE,
            size=CROP_SIZE,
            len_shift=2,
        )
    else:
        raise ValueError(f"Unknown MODEL_INDEX: {index}")
```

### 7.3 训练循环修改

最大改动是处理"模型返回 list（每 stage 输出）"的情况：

```python
def train_epoch(epoch, model, optimizer, ...):
    is_unfolding = MODEL_INDEX >= 7
    
    for step in range(num_iters):
        if is_unfolding:
            g, gt, mask3d, PhiPhiT = next(loader)
            g, gt, mask3d, PhiPhiT = g.cuda(), gt.cuda(), mask3d.cuda(), PhiPhiT.cuda()
            
            # 模型返回 list of K outputs
            outputs = model(g, input_mask=(mask3d, PhiPhiT))
            
            if MULTI_STAGE_LOSS and len(outputs) >= 4:
                # DPU-style multi-stage loss
                loss = (rmse(outputs[-1], gt) 
                        + 0.7 * rmse(outputs[-2], gt)
                        + 0.5 * rmse(outputs[-3], gt)
                        + 0.3 * rmse(outputs[-4], gt))
            else:
                # 单 stage loss (K<4 或 disable)
                loss = rmse(outputs[-1], gt)
            
            pred_for_metric = outputs[-1]  # 用最后一个 stage 计算 PSNR
        else:
            # 已有的非 unfolding 流程,完全不变
            input_meas, gt, mask3d, shift_mask = next(loader)
            ...
            pred = model(input_meas, shift_mask)
            loss = rmse(pred, gt)
            pred_for_metric = pred
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # PSNR/SSIM 等指标计算 (用 pred_for_metric)
        ...

def rmse(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt))
```

### 7.4 数据加载切换

```python
def main():
    is_unfolding = MODEL_INDEX >= 7
    
    # Dataset 设置
    train_set = MyDataset(..., return_unfolding_data=is_unfolding)
    test_set = MyDataset(..., return_unfolding_data=is_unfolding, isTrain=False)
    
    model = build_model(MODEL_INDEX).cuda()
    ...
```

### 7.5 注意 BATCH_SIZE 调整

unfolding 模型的内存占用比单 stage 大很多（每个 stage 都要保存中间激活用于 backward）。

经验值：
- NUM_STAGES=3 + SHARE_WEIGHTS=False: BATCH_SIZE=5 仍可
- NUM_STAGES=5 + SHARE_WEIGHTS=False: 建议 BATCH_SIZE=3
- NUM_STAGES=9 + SHARE_WEIGHTS=False: 建议 BATCH_SIZE=2
- NUM_STAGES=K + SHARE_WEIGHTS=True: BATCH_SIZE 可保持 5

参考 DPU/Train.py 第 26 行用 BATCH_SIZE=2。

---

## 8. 修改文件：test.py

### 8.1 推理流程

```python
# 加载 unfolding 模型
model = build_model(MODEL_INDEX).cuda()
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

# 数据加载切换为 unfolding 格式
test_set = MyDataset(..., return_unfolding_data=True, isTrain=False)

# 推理
with torch.no_grad():
    for i, (g, gt, mask3d, PhiPhiT) in enumerate(test_loader):
        g, mask3d, PhiPhiT = g.cuda(), mask3d.cuda(), PhiPhiT.cuda()
        outputs = model(g, input_mask=(mask3d, PhiPhiT))
        pred = outputs[-1]  # 取最后一个 stage 作为最终输出
        
        # 评估
        psnr_i = torch_psnr(pred[0], gt[0])
        ssim_i = torch_ssim(pred[0], gt[0])
        ...
```

### 8.2 可选: 中间 stage 可视化

unfolding 的一个有趣特性是可以观察重建质量随 stage 演化:

```python
# 推理时保留所有 stage 输出
outputs = model(g, input_mask=(mask3d, PhiPhiT))
psnr_per_stage = [torch_psnr(o[0], gt[0]) for o in outputs]
print(f"Scene {i}: PSNR per stage = {psnr_per_stage}")
# 期望: [27.x, 30.x, 33.x, 35.x, 37.x] 单调上升
```

这是论文中可以放进 figure 的可视化材料,展示 unfolding 的迭代精化效果。

---

## 9. 训练超参数配置建议

### 9.1 推荐配置矩阵

| 实验编号 | NUM_STAGES | SHARE_WEIGHTS | BATCH | 预期 PSNR | 训练时间(单 epoch) |
|---------|-----------|--------------|-------|----------|-----------------|
| Run-1 | 3 | False | 5 | ~36.5 | ~12 min |
| Run-2 | 5 | False | 3 | ~37.5 | ~20 min |
| Run-3 | 5 | True | 5 | ~36.5 | ~12 min |
| Run-4 | 9 | False | 2 | ~38.5 | ~35 min |
| Run-5 (主推) | 9 | True | 5 | ~37.8 | ~22 min |

**优先跑 Run-2 验证 unfolding 框架是否正常工作**。如果 PSNR 在 30 epoch 后达到 36+，说明实现正确，再扩展到 Run-4 跑长。

### 9.2 学习率与 scheduler

继承现有的 Cosine Annealing:

```python
LEARNING_RATE = 4e-4   # 与已有模型相同
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=1e-6)
```

DPU 和 SSR 也都用 4e-4 + Cosine。无需改动。

### 9.3 训练 epoch 数

unfolding 模型通常需要更多 epoch 收敛。建议:

- 单 stage 模型 300 epoch 已饱和
- unfolding 5stg: 训 350 epoch
- unfolding 9stg: 训 400 epoch

---

## 10. 关键陷阱与调试要点

### 10.1 PhiPhiT 计算细节(高优先级!)

**陷阱 1**: PhiPhiT 必须在 **shifted 空间**计算,不是 spatial 空间。

错误: `PhiPhiT = torch.sum(mask3d ** 2, dim=0)`  ← 这是 [H, W],错!
正确: 先 shift mask3d,再平方再 sum,结果是 [H, W'] 与 g 同形状

**陷阱 2**: PhiPhiT 中可能有 0(对应所有波段在该位置都被 mask 完全遮挡),除法会爆炸。
DPU 的处理: `Phi_s_batch[Phi_s_batch == 0] = 1`

**陷阱 3**: PhiPhiT 是 spatial 量,batch 内可以共享(因为 mask 固定)。但 DPU/SSR 在每个 batch item 都重复计算,简化代码。我们也跟进。

### 10.2 模型输出形状

unfolding 模型 forward 返回 **list**,不是 tensor。training/test 都要适配。

```python
# 错误:
loss = rmse(model(g, mask), gt)

# 正确:
outputs = model(g, mask)   # list of tensors
loss = rmse(outputs[-1], gt)  # 或 multi-stage loss
```

### 10.3 share_weights 模式的梯度

当 `share_weights=True`,所有 stage 调用同一个 `self.shared_prior`。PyTorch 自动处理梯度累积,不需要特殊操作。

但要注意: Adam 的 momentum/variance 是 per-parameter 的,不是 per-call 的。共享 prior 的参数会从 K 个 stage 累积梯度,等价于 batch_size × K。**实际学习率可能需要降低 K^0.5 倍**(平方根法则)。

实操: 先用相同 LR 跑,看是否发散。如果发散,降到 LR / sqrt(K)。

### 10.4 Checkpoint 兼容性

新模型的 state_dict 与已有 4 个模型不兼容(多了 rho_estimators 和 initial_conv)。
**Checkpoint 文件名需要包含 model_name 区分**:

```python
save_dir = f"result/model/{time_str}_{MODELS[MODEL_INDEX][1]}_stages{NUM_STAGES}/"
```

不要把 unfolding 的 checkpoint 误覆盖到已有 baseline。

### 10.5 数值稳定性

GD step 的 `(g - Phi_f) / PhiPhiT` 在训练初期可能很大(f 还没学好,残差大)。建议:

```python
residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
residual = residual.clamp(min=-10, max=10)  # 限制极端值
```

---

## 11. 验证清单与开发顺序

### 11.1 推荐开发顺序

**1: unfolding_ops.py**
- [ ] shift_batch / shift_back_batch 实现并测试(对比 DPU/Utils.py 的 shift_4/shift_back)
- [ ] 单元测试: shift 然后 shift_back 应该恢复原图(误差 < 1e-5)
- [ ] mul_Phi_f / mul_PhiT_residual 实现
- [ ] 单元测试: 数学性质 `<Φf, g> = <f, Φ^T g>`(伴随性)
- [ ] ParaEstimator 实现并跑通 forward

**2: dataset.py 修改**
- [ ] compute_PhiPhiT 实现并验证形状
- [ ] return_unfolding_data 切换逻辑
- [ ] 测试: print PhiPhiT.shape, .min(), .max() 看是否合理(min 应是 1.0,max 应在 5-15)

**3: wpo3d_unfold.py**
- [ ] WaveMST_3D_Unfold 实现
- [ ] **关键**: 先 view wpo3d.py 确认 prior network 接收什么 mask
- [ ] 测试: forward pass 一个 batch,确认输出 shape `[B, C, H, W]` × num_stages
- [ ] 测试: backward pass 不报 NaN

**4: train.py 修改**
- [ ] CONFIG 区扩展
- [ ] build_model 扩展
- [ ] 训练循环修改(unfolding 路径)
- [ ] 跑 1 个 epoch 看 loss 是否下降

**5: 完整训练 Run-2 (NUM_STAGES=5)**
- [ ] 训练 300 epoch
- [ ] 30 epoch 时 PSNR 应 > 33.0(说明在学)
- [ ] 100 epoch 时 PSNR 应 > 35.5
- [ ] 300 epoch 时 PSNR 应 > 37.0

**6**: 根据 Run-2 结果决定下一步:
- 如果 PSNR > 37: 跑 Run-4 (NUM_STAGES=9)
- 如果 PSNR < 36: 调试,可能是 GD step 实现错误

---

## 12. 对照参考:DPU 与 SSR 的关键代码片段

### 12.1 DPU 的 ADMM unfolding 循环(参考)

`DPU/Model.py` 第 374-385 行:

```python
out = []
for i in range(self.stage):
    mu = self.mu[i](f)
    z = self.net_stage[3*i](torch.cat([f + y/mu + r, f], dim=1))      # IPB
    r = self.net_stage[3*i+1](torch.cat([z_ori - y/mu - f, f], dim=1), # DPB
                              Phi, Phi_compressive)
    Phi_f = self.mul_Phif(Phi_shift, z - r - y/mu)
    f = z - r - y/mu + self.mul_PhiTg(
        Phi_shift, torch.div(g - Phi_f, mu + PhiPhiT))                # GD step
    f = self.net_stage[3*i+2](f, z - r)                                # Fusion
    z_ori = z
    y = y + mu * (f - z + r)                                           # Lagrange update
    out.append(f)
```

我们简化为(SSR 风格,无 Lagrange 乘子):

```python
out = []
for k in range(self.num_stages):
    rho_k = self.rho_estimators[k](f)
    Phi_f = mul_Phi_f(Phi_shift, f, ...)
    z = f + rho_k * mul_PhiT_residual(Phi_shift, (g - Phi_f) / PhiPhiT, ...)  # GD
    f = self.get_prior(k)(z, Phi)                                              # Prior
    out.append(f)
return out
```

### 12.2 SSR 的 GAP unfolding(我们直接参考的版本)

`SSR/Model.py` 第 356-368 行:

```python
out = []
for i in range(self.stage):
    '''LMP'''
    rho = self.rhos[i](f)
    Phi_f = self.mul_Phif(Phi_shift, f)
    z = f + rho * self.mul_PhiTg(Phi_shift, torch.div(g - Phi_f, PhiPhiT))
    '''SSRU'''
    f = self.net_stage[2 * i](z, Phi)
    '''ARB'''
    f = self.net_stage[2 * i + 1](f)
    out.append(f)
return out
```

我们的版本去掉 ARB(那是 SSR 特有的空间整流),只保留 LMP + Prior:

```python
out = []
for k in range(self.num_stages):
    rho_k = self.rho_estimators[k](f)
    Phi_f = mul_Phi_f(Phi_shift, f, ...)
    z = f + rho_k * mul_PhiT_residual(Phi_shift, (g - Phi_f) / PhiPhiT, ...)
    f = self.get_prior(k)(z, Phi)
    out.append(f)
return out
```

### 12.3 DPU 的 multi-stage loss(我们采用的版本)

`DPU/Train.py` 第 92-93 行:

```python
loss = loss_f(mse, out[opt.stage-1], label) + 0.7 * loss_f(mse, out[opt.stage-2], label) + \
       0.5 * loss_f(mse, out[opt.stage-3], label) + 0.3 * loss_f(mse, out[opt.stage-4], label)
```

我们的等价代码:

```python
if MULTI_STAGE_LOSS and len(outputs) >= 4:
    loss = (rmse(outputs[-1], gt)
            + 0.7 * rmse(outputs[-2], gt)
            + 0.5 * rmse(outputs[-3], gt)
            + 0.3 * rmse(outputs[-4], gt))
else:
    loss = rmse(outputs[-1], gt)
```

### 12.4 SSR 的 ParaEstimator 设计(我们参考)

`SSR/Model.py` 第 65-87 行,设计简洁:

```python
class Para_Estimator(nn.Module):
    def __init__(self, in_nc=28, out_nc=1, channel=32):
        super().__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Sequential(
            nn.Conv2d(channel, channel*2, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel*2, channel*4, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(channel*4, out_nc, 1, 1, 0, bias=True)
    
    def forward(self, x):
        x = self.fution(x)
        x = self.down_sample(x)
        x = self.avg_pool(x)
        x = self.out(x)
        return x   # 注意 SSR 没有 sigmoid, 而是让网络自由学习
```

我们的版本加 sigmoid 限制范围(更稳定):

```python
return torch.sigmoid(self.out(x))
```

如果训练发散,可以去掉 sigmoid 试试。

### 12.5 DPU 的 PhiPhiT 计算

`DPU/Dataset.py` 第 109-110 行:

```python
Phi_s_batch = torch.sum(shift_3(Phi_batch, 2) ** 2, 0)
Phi_s_batch[Phi_s_batch == 0] = 1
```

我们的等价代码:

```python
def compute_PhiPhiT(mask3d, len_shift=2):
    # mask3d: [C, H, W], not batched (per-sample in dataset.__getitem__)
    nC, H, W = mask3d.shape
    Phi_shifted_sq = torch.zeros(nC, H, W + (nC-1)*len_shift, device=mask3d.device)
    for c in range(nC):
        Phi_shifted_sq[c, :, c*len_shift:c*len_shift+W] = mask3d[c, :, :] ** 2
    PhiPhiT = torch.sum(Phi_shifted_sq, dim=0, keepdim=True)
    PhiPhiT[PhiPhiT == 0] = 1.0
    return PhiPhiT
```

---

## 附录:与已有代码的关系

### 不变的部分

完全不动的文件(已训练完成的模型仍可加载推理):
- `wpo3d.py` (WaveMST_3D / WaveMST_KG 模块本身)
- `wpo_smsa.py`, `wpo_mamba.py`, `mst.py`, `mask_ops.py`
- `loss.py`, `viz.py`
- `helmholtz_ops.py`, `wpo3d_phys.py`, `helm_pure.py`, `wpo3d_helm.py`(Model 4-6)

新增的文件:
- `unfolding_ops.py`(共用工具)
- `wpo3d_unfold.py`(Model 7,8 的本体)

### 论文叙事(交付 Claude Code 时仅供参考)

实现完成后,论文 contribution 表述:

> 我们提出基于阻尼波动方程的 deep unfolding 框架用于光谱压缩重建。每个 unfolding stage 把"测量约束(GD step)"和"波传播先验(3D-WPO/KG)"耦合,通过 K 次迭代逼近物理一致的重建。Klein-Gordon 版本进一步引入物理波数 $k(\lambda)=2\pi/\lambda$ 作为色散关系的硬先验,使光谱保真度(SAM)显著优于纯数据驱动方法。

**消融表预期**:

| 模型 | PSNR | SSIM | SAM | Params |
|------|------|------|-----|-------|
| 3D-WPO Pure (现状,非 unfolding) | 34.70 | 0.9432 | 0.1343 | 0.79M |
| 3D-WPO Unfold (5stg,share) | ~36.5 | ~0.95 | ~0.12 | 0.85M |
| 3D-WPO Unfold (5stg,no share) | ~37.5 | ~0.96 | ~0.11 | 4.0M |
| 3D-WPO Unfold (9stg,no share) | ~38.5 | ~0.965 | ~0.10 | 7.2M |
| 3D-KG Unfold (9stg,no share) | ~38.4 | ~0.965 | **~0.085** | 7.2M |

KG 版本的 SAM 显著低于 Pure 版本,这是物理波数注入的效果——可作为论文核心卖点。

