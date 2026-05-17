# Stage 2 物理增强模块 — Claude Code Handoff

> **前置状态**：5-stage 共享权重 unfolding 已跑通，100 epoch PSNR = 37.8 dB。
> **本次目标**：在现有 unfolding 框架上添加三个物理增强模块，全部以 train.py 的 CONFIG 软开关控制，可自由组合。
> **原则**：不修改 `wpo3d.py` 和 `mask_ops.py`（已验证的基础模块），只修改/新增外层包装。

---

## 目录

1. [三个增强模块概览](#1-三个增强模块概览)
2. [模块 A：源项注入（Source Injection）](#2-模块-a源项注入source-injection)
3. [模块 B：低秩 WPO（Low-Rank WPO）](#3-模块-b低秩-wpolow-rank-wpo)
4. [模块 C：色散介质（Dispersive Medium）](#4-模块-c色散介质dispersive-medium)
5. [train.py CONFIG 接口设计](#5-trainpy-config-接口设计)
6. [wpo3d_unfold.py 修改方案](#6-wpo3d_unfoldpy-修改方案)
7. [新增文件：enhancement_ops.py](#7-新增文件enhancement_opspy)
8. [损失函数扩展](#8-损失函数扩展)
9. [预期实验矩阵](#9-预期实验矩阵)
10. [关键陷阱与调试要点](#10-关键陷阱与调试要点)
11. [验证清单与开发顺序](#11-验证清单与开发顺序)
12. [时空复杂度优化方案](#12-时空复杂度优化方案)

---

## 1. 三个增强模块概览

| 模块 | 物理含义 | 作用位置 | 预期增益 | 额外参数 |
|------|---------|---------|---------|---------|
| **A: 源项注入** | 非齐次波方程，$\Phi^T g$ 作为持续光照源 | Prior step 输入拼接 | +0.3~0.8 dB | ~少量（1×1 conv） |
| **B: 低秩 WPO** | 光谱本征模截断，只传播前 r 个主模式 | WPO 频域调制内部 | +0.2~0.5 dB (SAM 提升明显) | ~少量（投影矩阵） |
| **C: 色散介质** | 空间依赖波速 $v_s(\mathbf{r})$，不同地物不同传播 | GD step 后的局部修正 | +0.2~0.5 dB | ~中等（小 CNN） |

三者完全正交，可任意组合。设计为 train.py 中的 bool 开关：

```python
USE_SOURCE_INJECTION = False   # 模块 A
USE_LOWRANK_WPO     = False   # 模块 B
USE_DISPERSIVE      = False   # 模块 C
LOWRANK_R           = 8       # 低秩截断的秩（模块 B 专用）
```

---

## 2. 模块 A：源项注入（Source Injection）

### 2.1 物理动机

当前 unfolding 的 prior step 只接收 GD step 后的 $z$：
$$f^{(k+1)} = \text{WPO3D}(z^{(k)}, \Phi)$$

但物理上，CASSI 系统有"持续光源"——场景反射光在每个位置注入能量。非齐次波方程：
$$\partial_{tt} u + \alpha \partial_t u = v^2 \nabla^2 u + S(\mathbf{r}, \lambda)$$

其中 $S = \beta \cdot \Phi^T g$ 是测量的反投影。这相当于在每个 unfolding stage 里，WPO prior 不仅看当前估计 $z$，还看一个"从测量中提取的物理源"。

### 2.2 数学形式

Duhamel 原理给出非齐次解的源项贡献：
$$\hat{u}_{\text{source}}(\omega) = \frac{1 - e^{-\alpha t/2}\cos(\omega_d t)}{\omega_0^2} \cdot \hat{S}(\omega)$$

低频源贡献大（$\omega_0^2$ 小），高频贡献小——天然低通。

### 2.3 工程实现（简化版）

**不修改 wpo3d.py 内部**，而是在 unfolding wrapper 中把源项作为额外输入拼接到 prior 的输入：

```python
# 在 wpo3d_unfold.py forward 循环中：
if use_source_injection:
    # 计算 Φ^T g（反投影测量值到 HSI 空间）
    PhiT_g = mul_PhiT_residual(Phi_shift, g, len_shift, size)  # [B, C, H, W]
    # 融合源项到 z
    z_with_source = self.source_conv(torch.cat([z, PhiT_g], dim=1))  # [B, C, H, W]
    f = self.get_prior(k)(z_with_source, Phi)
else:
    f = self.get_prior(k)(z, Phi)
```

需要新增：
- `self.source_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)`：每 stage 一个（或共享）

### 2.4 为什么不用 mask_ops.py 的 MaskSourceB？

MaskSourceB 是在 WPO **内部**做源项叠加（Duhamel 积分的完整形式）。这里我们用更简单的"输入拼接"——把源信息直接注入 WPO 的输入端。两者不冲突：

- MaskSourceB：WPO 内部物理源（修改频域调制结果）
- Source Injection：unfolding 外部信息注入（修改 WPO 的输入）

如果想同时启用两者，可以把 mask_mode 设为 'B'（MaskSourceB）加上 source_injection=True（外部注入），效果叠加。但默认只用外部注入（更简单，更容易调）。

### 2.5 参数量与显存影响

- 新增参数：每 stage 一个 `nn.Conv2d(56, 28, 1)` = 56×28 + 28 = 1,596 参数
- 5 stage 独立：+7,980 参数（可忽略）
- 显存：多一次 `mul_PhiT_residual` 计算（与 GD step 相同代价），每 stage +1 次 shift_back
- 训练速度：约慢 5-8%

---

## 3. 模块 B：低秩 WPO（Low-Rank WPO）

### 3.1 物理动机

HSI 数据的光谱维度具有强低秩性：28 个波段实际上由 5-10 个"光谱 endmember"线性组合。这对应亥姆霍兹方程的本征模展开：

$$u(\mathbf{r}, \lambda) = \sum_{n=1}^r c_n(\mathbf{r}) \cdot s_n(\lambda), \quad r \ll C$$

标准 3D-WPO 对所有 $C \times H \times (W/2+1)$ 个频域分量做同样的调制。低秩版本只让前 $r$ 个光谱主模式参与波传播，抑制噪声模式。

### 3.2 数学形式

设 WPO3D 的输入 $u_0 \in \mathbb{R}^{B \times C \times H \times W}$。

**光谱低秩投影**：
$$u_0^{\text{lr}} = B \cdot (A \cdot u_0)$$

其中 $A \in \mathbb{R}^{r \times C}$（压缩），$B \in \mathbb{R}^{C \times r}$（重建），$r=8$ 或 $12$。

操作上等价于：
1. 在光谱维做 `A @ u0`：[B, C, H, W] → [B, r, H, W]
2. 在低秩空间做 3D WPO（维度从 C=28 降到 r=8）
3. 用 `B @ out` 映射回 [B, C, H, W]

### 3.3 工程实现

**方案选择**：不修改 wpo3d.py 内部的 WPO3D 类。而是在 unfolding wrapper 中，在调用 prior 之前/之后加投影：

```python
if use_lowrank:
    # 低秩投影：[B, C, H, W] → [B, r, H, W]
    z_lr = self.proj_down[k](z)      # nn.Conv2d(C, r, 1)
    
    # 在低秩空间做 WPO（需要 dim=r 的 prior）
    # 方案 1：用单独的低秩 prior（dim=r）
    f_lr = self.lr_prior[k](z_lr, Phi_lr)
    
    # 投影回全秩：[B, r, H, W] → [B, C, H, W]
    f = self.proj_up[k](f_lr)        # nn.Conv2d(r, C, 1)
```

**但这需要新建 dim=r 的 WPO prior**，与已有 dim=28 的不共享。

**替代方案（推荐，更简单）**：保持 prior 的 dim=28 不变，在 WPO **输入端**加低秩瓶颈：

```python
if use_lowrank:
    # 瓶颈投影（在 prior 之前）
    z_proj = self.lr_bottleneck[k](z)  # Conv(C,r,1) → ReLU → Conv(r,C,1)
    f = self.get_prior(k)(z_proj, Phi)
else:
    f = self.get_prior(k)(z, Phi)
```

其中 `lr_bottleneck` 是：
```python
nn.Sequential(
    nn.Conv2d(dim, rank, 1, bias=False),   # [B,C,H,W] → [B,r,H,W]
    nn.ReLU(inplace=True),
    nn.Conv2d(rank, dim, 1, bias=False),   # [B,r,H,W] → [B,C,H,W]
)
```

这个瓶颈强制信息经过 rank-r 的通道，等效于对光谱做低秩滤波。物理上对应"只保留 r 个光谱本征模"。

### 3.4 为什么这能提升 SAM

SAM（Spectral Angle Mapper）衡量光谱形状保真度。噪声和过拟合通常表现为高阶光谱模式（rank > r 的分量）。低秩瓶颈天然滤除这些，保留主要光谱模式，因此 SAM 会改善。

### 3.5 参数量与显存影响

- `lr_bottleneck` 每个：$C \times r + r \times C = 2Cr$。C=28, r=8 → 448 参数/stage
- 5 stage：+2,240 参数（可忽略）
- 显存：增加一个瓶颈 forward（negligible）
- 训练速度：几乎无影响（< 1%）

### 3.6 r 的选择

| r | 物理含义 | 预期效果 |
|---|---------|---------|
| 4 | 极度压缩，只保留 4 个主模式 | SAM 最好，但 PSNR 可能略降 |
| 8 | 推荐默认值，保留主要信息 | SAM 和 PSNR 均有提升 |
| 12 | 轻度压缩 | 效果接近不压缩 |
| 28 | 等价于恒等（无效果） | 退化为标准 WPO |

---

## 4. 模块 C：色散介质（Dispersive Medium）

### 4.1 物理动机

标准 3D-WPO 假设均匀介质：波速 $v_s$ 是全局标量。但真实 HSI 中，不同地物（植被/水体/建筑）有不同的光学响应——对应不同的"有效折射率"。

色散介质波方程：
$$\partial_{tt} u + \alpha \partial_t u = \nabla \cdot (v^2(\mathbf{r}) \nabla u)$$

当 $v$ 依赖空间时，闭式解不再适用。但可以用**算子分裂**：
1. 用空间平均 $\bar{v}_s$ 做标准 WPO（已有）
2. 用局部修正 $\delta v_s(\mathbf{r})$ 做一阶 Born 修正

### 4.2 工程实现

Born 修正的形式：
$$f_{\text{dispersive}} = f_{\text{wpo}} + \gamma \cdot \text{LocalRefine}(f_{\text{wpo}}, \delta v)$$

其中 $\delta v(\mathbf{r})$ 由小 CNN 从当前特征预测（每个空间位置学一个"有效折射率偏差"）。

```python
if use_dispersive:
    # 预测空间依赖的折射率偏差
    delta_v = self.dispersion_net[k](f)        # [B, 1, H, W] 或 [B, C, H, W]
    
    # Born 修正：局部修正 = delta_v * Laplacian(f) 的近似
    # 用 DWConv 近似 Laplacian
    laplacian_f = self.laplacian_conv(f)       # [B, C, H, W]
    correction = delta_v * laplacian_f
    
    f = f + self.disp_weight[k] * correction
```

`dispersion_net` 设计：
```python
nn.Sequential(
    nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # DWConv 空间特征
    nn.ReLU(inplace=True),
    nn.Conv2d(dim, 1, 1, bias=True),                        # 压缩到单通道
    nn.Tanh(),                                               # 限制在 [-1, 1]
)
```

`laplacian_conv`：固定权重的 DWConv（Laplacian 模板 [[0,1,0],[1,-4,1],[0,1,0]]），不参与训练：
```python
# 初始化时
kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
kernel = kernel.view(1,1,3,3).repeat(dim, 1, 1, 1)
self.laplacian_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
self.laplacian_conv.weight = nn.Parameter(kernel, requires_grad=False)
```

### 4.3 作用位置

色散修正在 **prior step 之后**施加（对 WPO 输出做后处理修正）：

```python
# Prior step
f = self.get_prior(k)(z, Phi)

# 色散修正（在 prior 之后）
if use_dispersive:
    delta_v = self.dispersion_net[k](f)
    laplacian_f = self.laplacian_conv(f)
    f = f + self.disp_weight[k] * delta_v * laplacian_f
```

### 4.4 参数量与显存影响

- `dispersion_net` 每个：$C \times 9 + C + 1 \times C + 1 = 28 \times 9 + 28 + 28 + 1 = 309$ 参数/stage
- `disp_weight`：1 参数/stage
- `laplacian_conv`：固定，0 可训练参数
- 5 stage：~1,550 可训练参数
- 显存：一次 DWConv + 一次固定卷积 per stage（negligible）
- 训练速度：< 2% 影响

### 4.5 与 KG 方程的关系

KG 方程已经有波长依赖的 $k^2(\lambda)$（通过 MaskKleinGordonD 实现）。色散介质加入的是**空间依赖**的修正。两者正交：

- KG：不同波长有不同固有频率（光谱方向异质性）
- 色散：不同空间位置有不同波速（空间方向异质性）

可以同时启用（Model 8 + USE_DISPERSIVE=True）。

---

## 5. train.py CONFIG 接口设计

### 5.1 新增配置项

```python
# ════════════════════════════════════════════
# CONFIG（在此修改所有超参数）
# ════════════════════════════════════════════
MODEL_INDEX  = 7       # 0: WaveMST_3D  1: WaveMST_KG  7: Unfold_3D  8: Unfold_KG
GPU_ID       = '0'
BATCH_SIZE   = 2
MAX_EPOCH    = 300
LR           = 4e-4
# ... 现有配置保持不变 ...

# Unfolding 专用配置
NUM_STAGES          = 5
SHARE_STAGE_WEIGHTS = True
MULTI_STAGE_LOSS    = True

# ★ 物理增强模块（仅 MODEL_INDEX >= 7 时生效，可任意组合）
USE_SOURCE_INJECTION = False   # 模块 A：Φ^T g 源项注入到 prior 输入
USE_LOWRANK_WPO     = False   # 模块 B：低秩光谱瓶颈
USE_DISPERSIVE      = False   # 模块 C：空间色散修正
LOWRANK_R           = 8       # 模块 B 的截断秩 r（4/8/12）
# ════════════════════════════════════════════
```

### 5.2 build_model 传参

```python
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
        # 新增参数：
        use_source_injection=USE_SOURCE_INJECTION,
        use_lowrank=USE_LOWRANK_WPO,
        use_dispersive=USE_DISPERSIVE,
        lowrank_r=LOWRANK_R,
    )
```

Model 8 同理，额外传 `use_kg=True`。

### 5.3 test.py 对应更新

test.py 的 CONFIG 区同样加上这四个开关，确保与训练一致：

```python
USE_SOURCE_INJECTION = False
USE_LOWRANK_WPO     = False
USE_DISPERSIVE      = False
LOWRANK_R           = 8
```

---

## 6. wpo3d_unfold.py 修改方案

### 6.1 __init__ 扩展

```python
class WaveMST_3D_Unfold(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=None,
                 num_stages=5, share_weights=False, use_kg=False,
                 mask_mode='A', size=256, len_shift=2,
                 # ★ 新增：
                 use_source_injection=False,
                 use_lowrank=False,
                 use_dispersive=False,
                 lowrank_r=8):
        super().__init__()
        # ... 现有初始化保持不变 ...
        
        self.use_source_injection = use_source_injection
        self.use_lowrank = use_lowrank
        self.use_dispersive = use_dispersive
        
        # 模块 A：源项融合卷积
        if use_source_injection:
            self.source_convs = nn.ModuleList([
                nn.Conv2d(dim * 2, dim, 1, 1, 0)
                for _ in range(num_stages)
            ])
        
        # 模块 B：低秩瓶颈
        if use_lowrank:
            self.lr_bottlenecks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim, lowrank_r, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(lowrank_r, dim, 1, bias=False),
                )
                for _ in range(num_stages)
            ])
        
        # 模块 C：色散修正
        if use_dispersive:
            from enhancement_ops import DispersionCorrector
            self.dispersion_corrs = nn.ModuleList([
                DispersionCorrector(dim)
                for _ in range(num_stages)
            ])
```

### 6.2 forward 循环修改

```python
def forward(self, g, input_mask):
    Phi, PhiPhiT = input_mask
    Phi_shift = shift_batch(Phi, self.len_shift)
    
    # 初始化 f0（不变）
    g_normal = g / self.nC * 2
    temp_g = g_normal.repeat(1, self.nC, 1, 1)
    f0 = shift_back_batch(temp_g, self.len_shift, self.size)
    f = self.initial_conv(torch.cat([f0, Phi], dim=1))
    
    # 预计算 Φ^T g（模块 A 用）
    if self.use_source_injection:
        PhiT_g = mul_PhiT_residual(Phi_shift, g, self.len_shift, self.size)
    
    outputs = []
    for k in range(self.num_stages):
        # ── GD step（不变）──
        rho_k = self.rho_estimators[k](f)
        Phi_f = mul_Phi_f(Phi_shift, f, self.len_shift)
        residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
        residual = residual.clamp(min=-10, max=10)
        z = f + rho_k * mul_PhiT_residual(
            Phi_shift, residual, self.len_shift, self.size
        )
        
        # ── 模块 A：源项注入 ──
        if self.use_source_injection:
            z = self.source_convs[k](torch.cat([z, PhiT_g], dim=1))
        
        # ── 模块 B：低秩瓶颈 ──
        if self.use_lowrank:
            z = self.lr_bottlenecks[k](z)
        
        # ── Prior step（WPO3D）──
        f = self.get_prior(k)(z, Phi)
        
        # ── 模块 C：色散修正 ──
        if self.use_dispersive:
            f = self.dispersion_corrs[k](f)
        
        outputs.append(f)
    
    return outputs
```

### 6.3 模块插入位置的逻辑

```
GD step → z
         ↓
[模块 A] 源项注入：z = conv([z, Φ^T g])  ← 在 prior 之前，给 prior 额外信息
         ↓
[模块 B] 低秩瓶颈：z = bottleneck(z)     ← 在 prior 之前，滤波输入
         ↓
Prior step (WPO3D)：f = WPO(z, Phi)
         ↓
[模块 C] 色散修正：f = f + γ·δv·∇²f      ← 在 prior 之后，物理后处理
         ↓
输出 f → 下一个 stage
```

---

## 7. 新增文件：enhancement_ops.py

```python
"""
enhancement_ops.py — Stage 2 物理增强模块

包含：
  - DispersionCorrector：空间色散 Born 修正（模块 C）
  
模块 A 和 B 直接用 nn.Conv2d/nn.Sequential，无需额外类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DispersionCorrector(nn.Module):
    """空间色散 Born 一阶修正。
    
    预测空间依赖的折射率偏差 δv(r)，用 Laplacian(f) 做修正：
        f_out = f + weight * δv(r) * Laplacian(f)
    
    物理含义：不同空间位置有不同波传播速度。
    """
    
    def __init__(self, dim=28):
        super().__init__()
        # 预测 δv(r)：空间依赖的折射率偏差
        self.delta_v_net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # DWConv
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 1, bias=True),   # 压缩到单通道
            nn.Tanh(),                          # 限制范围 [-1, 1]
        )
        # 可学习修正强度
        self.weight = nn.Parameter(torch.tensor(0.1))
        
        # 固定 Laplacian 卷积核（不参与训练）
        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.register_buffer('laplacian_kernel', kernel)
        self.dim = dim
    
    def forward(self, f):
        """
        f: [B, C, H, W]
        返回: [B, C, H, W] 修正后的 f
        """
        # Laplacian（固定卷积，无梯度到权重）
        laplacian_f = F.conv2d(f, self.laplacian_kernel, padding=1, groups=self.dim)
        
        # 空间依赖折射率偏差
        delta_v = self.delta_v_net(f)  # [B, 1, H, W]
        
        # Born 修正
        correction = delta_v * laplacian_f   # 广播 [B,1,H,W] * [B,C,H,W]
        return f + self.weight * correction
```

---

## 8. 损失函数扩展

### 8.1 现有损失（不变）

```python
def multi_stage_loss(outputs, gt):
    K = len(outputs)
    loss = rmse_loss(outputs[-1], gt)
    if K >= 2: loss += 0.7 * rmse_loss(outputs[-2], gt)
    if K >= 3: loss += 0.5 * rmse_loss(outputs[-3], gt)
    if K >= 4: loss += 0.3 * rmse_loss(outputs[-4], gt)
    return loss
```

### 8.2 可选：SAM 辅助损失（配合低秩模块 B）

当启用低秩 WPO 时，可以额外加一个 SAM 损失强化光谱保真：

```python
# train.py CONFIG
USE_SAM_LOSS = False   # 仅在 USE_LOWRANK_WPO=True 时推荐
SAM_LOSS_WEIGHT = 0.1

# 在 train_epoch 中：
if MULTI_STAGE_LOSS:
    loss = multi_stage_loss(outputs, gt)
else:
    loss = rmse_loss(outputs[-1], gt)

if USE_SAM_LOSS:
    from loss import torch_sam
    sam_val = torch_sam(outputs[-1], gt)
    loss = loss + SAM_LOSS_WEIGHT * sam_val
```

这个是可选的，如果发现低秩模块对 PSNR 有轻微降低但 SAM 大幅改善，可以用 SAM loss 平衡。

---

## 9. 预期实验矩阵

基于当前 baseline（5stg, share=True, 100ep → 37.8 dB）扩展：

| 编号 | 源项(A) | 低秩(B) | 色散(C) | r | 预期 PSNR | 预期 SAM 变化 | 说明 |
|:----:|:-------:|:-------:|:-------:|:-:|:---------:|:------------:|------|
| E0 | - | - | - | - | 37.8 | baseline | 当前 baseline |
| E1 | **ON** | - | - | - | ~38.2 | -5% | 单开源项 |
| E2 | - | **ON** | - | 8 | ~37.9 | **-15%** | 单开低秩（SAM 主要改善） |
| E3 | - | - | **ON** | - | ~38.0 | -3% | 单开色散 |
| E4 | ON | ON | - | 8 | ~38.4 | -18% | A+B 组合 |
| E5 | ON | - | ON | - | ~38.3 | -7% | A+C 组合 |
| E6 | ON | ON | ON | 8 | ~38.5 | -20% | 全开 |
| E7 | ON | ON | ON | 4 | ~38.3 | **-25%** | 全开+强压缩（SAM 最优） |

> 注：SAM 变化为相对于 baseline 的百分比下降（负号=改善）。

**推荐实验顺序**：E1 → E2 → E4 → E6

先跑 E1 验证源项注入有效，再验证低秩，最后组合。

---

## 10. 关键陷阱与调试要点

### 10.1 源项注入（模块 A）

**陷阱 1**：`PhiT_g` 只需计算一次（在循环外），不要在每个 stage 重复计算。它是常量。

**陷阱 2**：`mul_PhiT_residual(Phi_shift, g, ...)` 中传的是原始 g [B,1,H,W']，不是 residual。确保 g 的维度正确（不要误传 g_normal）。

**陷阱 3**：source_conv 的输出维度必须是 dim（28），不是 dim*2。检查 `nn.Conv2d(dim * 2, dim, 1)` 的输入确实是 [z, PhiT_g] concat 后的 56 通道。

### 10.2 低秩瓶颈（模块 B）

**陷阱 1**：`lr_bottleneck` 的两层 Conv2d 都是 `bias=False`——加 bias 会破坏低秩约束的物理意义（bias 相当于加了一个全秩的常数项）。

**陷阱 2**：r=4 时信息瓶颈很窄。如果 PSNR 大幅下降（>0.5 dB），说明 r 太小，先用 r=8 或 12。

**陷阱 3**：ReLU 在瓶颈中间是必要的——没有非线性，两个线性层的乘积仍是线性的（退化为单层），低秩约束无效。

### 10.3 色散修正（模块 C）

**陷阱 1**：Laplacian 卷积核必须是固定的（`requires_grad=False`），否则网络会学到非物理的"伪 Laplacian"。用 `register_buffer` 存储。

**陷阱 2**：`delta_v_net` 输出用 Tanh 限制在 [-1, 1]。如果初始化太大会让修正过强，导致训练初期不稳定。初始化 `self.weight = 0.1` 是保守设定。

**陷阱 3**：Laplacian 对边界敏感。用 `padding=1` 保持尺寸不变，边界值被 zero-pad，可能引入伪边界效应。如果发现边界区域 PSNR 低于中心区域，考虑改用 `reflect` padding：
```python
f_pad = F.pad(f, [1,1,1,1], mode='reflect')
laplacian_f = F.conv2d(f_pad, self.laplacian_kernel, groups=self.dim)
```

### 10.4 组合使用时的顺序

三个模块的执行顺序已在 §6.2 确定：A → B → Prior → C。
- A 在 prior 之前：给 WPO 更多信息
- B 在 prior 之前：滤波输入
- C 在 prior 之后：物理修正输出

**不要调换 A 和 B 的顺序**——如果先做低秩再拼接源项，源项信息会被瓶颈截断。

### 10.5 显存预估

| 配置 | 额外显存开销（5stg, batch=2） |
|------|:---------------------------:|
| 仅 A | ~+0.3 GB |
| 仅 B | ~+0.1 GB |
| 仅 C | ~+0.2 GB |
| A+B+C | ~+0.5 GB |

在 24GB GPU 上，当前 baseline（5stg, share=True, batch=2）约 12 GB，全开三个模块约 12.5 GB，完全可行。

---

## 11. 验证清单与开发顺序

### Phase 1：enhancement_ops.py（1 小时）

- [ ] 创建文件，实现 `DispersionCorrector`
- [ ] 验证 Laplacian 卷积核正确（对常数输入结果为 0，对二次函数输入结果为常数）
- [ ] 验证 forward 输入输出形状一致

### Phase 2：wpo3d_unfold.py 扩展（2 小时）

- [ ] 添加三个 bool 参数到 `__init__`
- [ ] 添加对应的 ModuleList
- [ ] 修改 forward 循环（按 A→B→Prior→C 顺序）
- [ ] 验证：三个开关全 False 时行为与现有代码完全相同（回归测试）
- [ ] 验证：单开每个模块时 forward pass 不报错

### Phase 3：train.py / test.py 更新（30 分钟）

- [ ] CONFIG 区添加四个新参数
- [ ] build_model 传参
- [ ] test.py 同步

### Phase 4：快速训练验证（每个约 30 epoch）

- [ ] E1（源项）：30 epoch 后 PSNR 是否 > baseline 同期
- [ ] E2（低秩）：30 epoch 后 SAM 是否改善
- [ ] E3（色散）：30 epoch 后 PSNR 是否改善

### Phase 5：组合实验

- [ ] E4/E6：全开组合，确认无冲突
- [ ] 长训练（300 epoch）跑出最终数字

---

## 12. 时空复杂度优化方案

> 本章讨论在**不丢失精度或仅以微小精度为代价**的前提下，现有代码的时空复杂度优化。

### 12.1 当前瓶颈分析

通过 OOM 报错位置（`wpo3d.py:144 _wave_modulate`）和内存占用观察，主要瓶颈为：

| 瓶颈 | 来源 | 占用 |
|------|------|------|
| 3D rFFT 中间张量 | `u0_fft`, `out_fft` 各 [B,C,H,W//2+1] complex64 | ~2× 实数存储 |
| K stage 反向传播 | 每 stage 保存完整激活图用于 backward | K × 单 stage 显存 |
| shift 操作的临时张量 | `shift_batch` 创建 [B,C,H,W'] 零张量 | W'=W+54 > W |
| ParaEstimator | 每 stage 独立的 down_sample + avg_pool | 小，但有 K 份 |

### 12.2 优化 1：梯度检查点（Gradient Checkpointing）

**原理**：不保存中间 stage 的激活图，反向传播时重新计算。用时间换空间。

**影响**：
- 显存：降至约 2× 单 stage（而非 K× 单 stage）
- 速度：训练慢约 30-40%（每 stage 的 forward 被执行两次）
- 精度：**完全无损**（数学等价）

**实现**：

```python
from torch.utils.checkpoint import checkpoint

def forward(self, g, input_mask):
    # ... 初始化部分不变 ...
    
    outputs = []
    for k in range(self.num_stages):
        if self.training and self.use_checkpoint:
            f = checkpoint(self._single_stage, f, g, Phi_shift, PhiPhiT, k,
                          use_reentrant=False)
        else:
            f = self._single_stage(f, g, Phi_shift, PhiPhiT, k)
        outputs.append(f)
    return outputs

def _single_stage(self, f, g, Phi_shift, PhiPhiT, k):
    """单 stage 的完整计算，用于 checkpoint"""
    rho_k = self.rho_estimators[k](f)
    Phi_f = mul_Phi_f(Phi_shift, f, self.len_shift)
    residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
    residual = residual.clamp(min=-10, max=10)
    z = f + rho_k * mul_PhiT_residual(Phi_shift, residual, self.len_shift, self.size)
    
    # 增强模块...
    
    f = self.get_prior(k)(z, Phi)
    return f
```

**CONFIG 新增**：
```python
USE_CHECKPOINT = True   # 梯度检查点（降显存，训练慢 30%）
```

**推荐场景**：
- 9 stage + 独立权重 + batch=2 仍 OOM → 开 checkpoint
- 5 stage + share=True + batch=2 够用 → 不需要开

### 12.3 优化 2：混合精度训练（AMP）

**原理**：用 float16 做大部分计算，仅在容易溢出的位置（loss、norm）用 float32。

**影响**：
- 显存：降约 30-40%（激活图减半）
- 速度：快约 20-30%（Tensor Core 加速）
- 精度：通常无损或 < 0.05 dB 差异

**实现**：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for step in range(batch_num):
    gt = shuffle_crop(...)
    
    with autocast():
        if IS_UNFOLDING:
            outputs = model(g, input_mask=(mask3d_batch, PhiPhiT))
            loss = multi_stage_loss(outputs, gt)
        else:
            pred = model(meas, shift_mask_train)
            loss = rmse_loss(pred, gt)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**注意事项**：
- 3D FFT 在 float16 下精度略有下降。如果发现训练不稳定，可对 `_wave_modulate` 单独用 float32：
  ```python
  @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
  def _wave_modulate(self, ...):
      ...
  ```
- `PhiPhiT.clamp(min=1e-6)` 在 float16 下 1e-6 接近下溢，改为 `clamp(min=1e-4)`

**CONFIG 新增**：
```python
USE_AMP = False   # 混合精度（降显存 + 加速，推荐 24GB 以下 GPU）
```

### 12.4 优化 3：shift 操作向量化

**当前实现**：
```python
def shift_batch(f, len_shift=2):
    B, C, H, W = f.shape
    shifted = torch.zeros(B, C, H, W + pad_w, ...)
    for c in range(C):  # ← 28 次循环！
        shifted[:, c, :, c*len_shift:c*len_shift+W] = f[:, c, :, :]
    return shifted
```

这个 for 循环在 GPU 上是 28 次 kernel launch，效率低。

**优化方案：用 torch.nn.functional.pad + 索引**：

```python
def shift_batch_fast(f, len_shift=2):
    """用 pad + gather 代替逐通道循环"""
    B, C, H, W = f.shape
    pad_w = (C - 1) * len_shift
    # 先 pad 右边
    f_padded = F.pad(f, [0, pad_w])  # [B, C, H, W+pad_w]
    # 用 roll 逐通道移位（仍是循环，但可以用 scatter 优化）
    # 更好：构造 index tensor 一次性 gather
    W_out = W + pad_w
    idx = torch.arange(W, device=f.device).unsqueeze(0).expand(C, -1)  # [C, W]
    offsets = (torch.arange(C, device=f.device) * len_shift).unsqueeze(1)  # [C, 1]
    idx = idx + offsets  # [C, W] 每个通道的目标起始位置
    
    shifted = torch.zeros(B, C, H, W_out, device=f.device, dtype=f.dtype)
    # scatter 方式
    idx_expanded = idx.unsqueeze(0).unsqueeze(2).expand(B, -1, H, -1)  # [B, C, H, W]
    shifted.scatter_(3, idx_expanded, f)
    return shifted
```

**影响**：
- 速度：减少 kernel launch overhead，大约快 20-30%
- 精度：**完全无损**
- 实测必要性：只有在 batch_num 很大（5000/batch）时收益明显

### 12.5 优化 4：PhiPhiT 预计算缓存

**当前问题**：`gen_meas_unfolding` 在每个 batch 的每个样本重复计算 PhiPhiT。但 mask 是固定的，PhiPhiT 只取决于 mask，是常量。

**优化**：在训练开始时预计算一次，缓存为 tensor：

```python
# main() 中，数据加载后：
PhiPhiT_cached = compute_PhiPhiT(mask3d, len_shift=2).cuda()  # [1, H, W']

# train_epoch 中：
PhiPhiT = PhiPhiT_cached.expand(B, -1, -1, -1)  # 无额外显存（expand 是 view）
```

**影响**：
- 速度：节省每 batch B 次的 PhiPhiT 计算（约 5% 训练时间）
- 显存：无影响（PhiPhiT 本来就存在）
- 精度：**完全无损**

**注意**：`gen_meas_unfolding` 仍然需要为每个样本计算 g（因为 gt 每次不同），但 PhiPhiT 部分可以提取出来。

### 12.6 优化 5：WPO3D 内部 FFT 尺寸优化

**背景**：cuFFT 对 2 的幂次长度最高效。当前 crop_size=256（2^8），已经最优。但光谱维 C=28 不是 2 的幂。

**优化**：对光谱维 pad 到 32（2^5）再做 FFT：

```python
# 在 _wave_modulate 中：
C_pad = 32  # 下一个 2^n
u0_padded = F.pad(u0, [0, 0, 0, 0, 0, C_pad - C])  # 光谱维 pad 到 32
u0_fft = torch.fft.rfftn(u0_padded, dim=(-3, -2, -1))
# ... 调制 ...
out = torch.fft.irfftn(out_fft, s=(C_pad, H, W), dim=(-3, -2, -1))
out = out[:, :C, :, :]  # 截取回 28
```

**影响**：
- 速度：cuFFT 对 32 比 28 快约 15-20%
- 显存：pad 4 个通道，略增（<5%）
- 精度：**完全无损**（FFT 对 zero-pad 是精确的）

**注意**：这需要修改 `wpo3d.py`，与"不修改 wpo3d.py"原则冲突。如果严格遵守，跳过此优化。如果允许小改动，收益可观。

### 12.7 优化 6：share_weights + LoRA 分支（精度-参数折中）

**背景**：share_weights=True 省参数但表达力受限。独立权重参数量 ×K。LoRA 是折中：

$$\text{WPO}^{(k)} = \text{WPO}_{\text{shared}} + \Delta^{(k)}, \quad \Delta^{(k)} = B_k A_k$$

每 stage 只有低秩修正 $\Delta^{(k)}$（LoRA rank=4/8），参数增加 $K \times 2 \times \text{dim} \times r$。

**影响**：
- 参数：share=True (0.85M) + K×LoRA → 约 0.9M（远小于独立的 4M）
- 精度：接近独立权重（差 0.2-0.5 dB），远优于纯共享
- 显存：与 share=True 相同（因为 base model 只有一份）

**实现思路**：在 `WPO3DBlock` 的关键层（phi/psi 的 Linear，FFN 的第一层）加 LoRA adapter。

**注意**：这需要修改 `wpo3d.py` 内部结构，属于较大改动。可作为 Stage 3 考虑。

### 12.8 优化总结与推荐

| 优化 | 精度影响 | 显存 | 速度 | 修改量 | 推荐度 |
|------|:-------:|:----:|:----:|:------:|:------:|
| 梯度检查点 | 无损 | -50% | -30% | 小 | ★★★★★ |
| 混合精度 AMP | < 0.05 dB | -35% | +25% | 小 | ★★★★ |
| shift 向量化 | 无损 | 不变 | +20% | 中 | ★★★ |
| PhiPhiT 缓存 | 无损 | 不变 | +5% | 小 | ★★★★★ |
| FFT pad 到 2^n | 无损 | +3% | +15% | 需改 wpo3d | ★★★ |
| LoRA 分支 | -0.2~0.5 dB | -40% | 不变 | 大 | ★★（Stage 3） |

**首选组合**：梯度检查点 + AMP + PhiPhiT 缓存。这三个改动小、无精度损失（或极小损失），可以让 9-stage 独立权重在 24GB GPU 上以 batch=3 训练。

**CONFIG 新增（完整）**：
```python
# 时空优化配置
USE_CHECKPOINT = False    # 梯度检查点（推荐 9stg 时开启）
USE_AMP        = False    # 混合精度（推荐 24GB 以下 GPU 开启）
CACHE_PHIPHIT  = True     # 预缓存 PhiPhiT（无代价，默认开）
```

---

## 附录：完整 CONFIG 区模板

```python
# ════════════════════════════════════════════
# CONFIG（在此修改所有超参数）
# ════════════════════════════════════════════
MODEL_INDEX  = 7       # 7: Unfold_3D  8: Unfold_KG
GPU_ID       = '0'
BATCH_SIZE   = 2
MAX_EPOCH    = 300
LR           = 4e-4
SCHEDULER    = 'CosineAnnealingLR'
MILESTONES   = [50, 100, 150, 200, 250]
EPOCH_SAMPLE = 5000
CROP_SIZE    = 256
NUM_BANDS    = 28
DIM          = 28
STAGE        = 2
NUM_BLOCKS   = [2, 2, 2]
MASK_MODE    = 'A'
INPUT_SETTING = 'H'
SAVE_THRESH  = 28.0

# Unfolding 配置
NUM_STAGES          = 5
SHARE_STAGE_WEIGHTS = True
MULTI_STAGE_LOSS    = True

# ★ 物理增强模块（可任意组合）
USE_SOURCE_INJECTION = False   # A: Φ^T g 源项注入
USE_LOWRANK_WPO     = False   # B: 低秩光谱瓶颈
USE_DISPERSIVE      = False   # C: 空间色散修正
LOWRANK_R           = 8       # B 的截断秩

# ★ 时空优化
USE_CHECKPOINT = False         # 梯度检查点
USE_AMP        = False         # 混合精度
CACHE_PHIPHIT  = True          # PhiPhiT 预缓存

# 辅助损失
USE_SAM_LOSS    = False        # SAM 辅助损失
SAM_LOSS_WEIGHT = 0.1

# 数据路径
DATA_ROOT  = Path('../dataset')
TRAIN_PATH = DATA_ROOT / 'CAVE_1024_npy'
TRAIN_PATH_FALLBACK = DATA_ROOT / 'CAVE_1024' / 'cave_1024_28'
TEST_PATH  = DATA_ROOT / 'TSA_simu_data' / 'Truth'
MASK_PATH  = DATA_ROOT / 'TSA_simu_data'
RESULT_ROOT = Path('./result')
# ════════════════════════════════════════════
```
