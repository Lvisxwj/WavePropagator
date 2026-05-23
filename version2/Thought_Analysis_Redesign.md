# 重新审视 CASSI HSI 重建：从问题本质出发的架构重设计

> **本文性质**：严肃的技术反思与方向修正。基于实验数据的冷静分析，而非"讲故事说服自己"。
>
> **核心结论预告**：之前的 ML 层堆砌方向是错的——不是因为 ML 不好，而是因为放错了位置。3D-WPO 的真正问题不是"特征学习能力不足"，而是"输入质量太差"。解决方案不是在 WPO 旁边堆 ML，而是在 WPO 之前做退化估计和初始场净化。

---

## 目录

1. [实验数据的冷静分析](#1-实验数据的冷静分析)
2. [之前的方向为什么错了](#2-之前的方向为什么错了)
3. [回到问题本质：3D-WPO 到底做了什么](#3-回到问题本质3d-wpo-到底做了什么)
4. [3D-WPO 对 HSI 重建的真实瓶颈](#4-3d-wpo-对-hsi-重建的真实瓶颈)
5. [SOTA 方法的退化处理范式](#5-sota-方法的退化处理范式)
6. [新范式设计：净化-传播-精化](#6-新范式设计净化-传播-精化)
7. [具体模块筛选与评估](#7-具体模块筛选与评估)
8. [降维与参数效率](#8-降维与参数效率)
9. [最终推荐架构](#9-最终推荐架构)
10. [实验路线图](#10-实验路线图)

---

## 1. 实验数据的冷静分析

### 1.1 ML 层实验的真实数据

| 配置 | 参数量 | PSNR@10ep | PSNR@30ep | PSNR@60ep | 训练时间/ep |
|------|-------|-----------|-----------|-----------|-----------|
| **纯 WPO (baseline)** | **0.79M** | **30.19** | **32.20** | **33.08** | **415s** |
| WSSA + symmetric | 3.80M (4.8×) | 30.87 | 33.24 | 34.24 | 590s (1.4×) |
| FBA + symmetric | 4.47M (5.7×) | 30.50 | 32.78 | 33.92 | 780s (1.9×) |
| WSSA + asymmetric | 3.67M (4.6×) | 30.75 | 32.92 | 34.24 | 540s (1.3×) |
| WSSA + alternating | 3.67M (4.6×) | 30.57 | 33.06 | 34.23 | 518s (1.3×) |

### 1.2 冷静的结论

**ML 层有一定效果，但参数效率很差**。以最好的 WSSA+symmetric 为例，用了 4.8 倍参数（0.79M→3.80M），换来的提升：

- @10ep：+0.68 dB（30.19→30.87）
- @30ep：+1.04 dB（32.20→33.24）
- @60ep：+1.16 dB（33.08→34.24）
- 推算 @300ep：约 +0.5~0.8 dB（34.70→35.2~35.5）

提升是真实的，但代价是参数量翻了近 5 倍，训练时间增加 40%。换算成"每增加 1M 参数带来的 dB 提升"（@60ep）：约 0.39 dB/M。

**FBA 表现最差**。4.47M 参数（最大），@60ep 只有 33.92（仅比 baseline 多 0.84 dB），训练时间近乎翻倍。小波分解与 WPO 的 FFT 功能重叠，增加了冗余计算而非互补信息。

**WSSA 的三种 U-Net 配置差异不大**。symmetric（34.24）、asymmetric（34.24）、alternating（34.23）在 @60ep 几乎持平，说明 U-Net 骨架的变化对性能影响很小——瓶颈不在架构设计上。

### 1.3 对比 unfolding 的效果

| 改进方式 | 额外参数 | PSNR 提升(@60ep) | 性价比(dB/M) |
|---------|---------|-----------------|------------|
| 5-stage unfolding | ~0.1M（ParaEstimator）| **+3.1 dB**（33.08→36.18）| **31 dB/M** |
| WSSA+symmetric | +3.01M | +1.16 dB（33.08→34.24）| 0.39 dB/M |
| FBA+symmetric | +3.68M | +0.84 dB（33.08→33.92）| 0.23 dB/M |

**Unfolding 的性价比是 ML 堆砌的约 80 倍。** ML 层的提升确实存在（~1 dB），但远不如 unfolding 的结构性改进（~3 dB）。更关键的是，unfolding 几乎不增加参数，而 ML 层增加了 4-5 倍参数。

这意味着：**ML 层不是完全无用，但它被放在了错误的位置**——在一个脏的初始场上做精细特征建模，收益被退化噪声严重稀释。如果先解决初始场质量问题（退化估计），ML 层的 1 dB 潜力才能被真正释放。

---

## 2. 之前的方向为什么错了

### 2.1 错误的诊断

之前的推理链条：

> "纯 WPO 38.21 dB 距 SOTA 差 2 dB → 瓶颈在特征学习能力不足 → 加 ML 层增强特征学习"

这个诊断**表面上合理但实际上错了**。正确的诊断应该是：

> "纯 WPO 38.21 dB 距 SOTA 差 2 dB → 差距来自什么？"

看 DPU 的消融（Table 3）：

| 组件 | PSNR | 增益来源 |
|------|------|---------|
| Baseline-1（无 FA，无 DPF）| 37.28 | — |
| + L/Swin-FA（注意力增强）| 38.49 | **+1.21**（注意力机制）|
| + Intuitive DPF | 38.76 | **+0.27**（直觉残差学习）|
| + Basic DPF | 39.23 | **+0.47**（数据保真公式化）|
| + Full DPF（双先验融合）| 39.62 | **+0.39**（融合）|

**退化建模（DPF）贡献了 1.13 dB**——和注意力机制的 1.21 dB 几乎相当。我们完全忽略了退化建模这一半的贡献。

### 2.2 正确的诊断

3D-WPO 的问题不是"特征学习能力不足"——它在频域的全局传播能力其实很强。问题是**它接收的输入太脏了**。

WPO 做的事情：从初始场 $u_0$（由 $\Phi^T g$ 经过简单 Conv 得到）出发，做物理传播。但 $\Phi^T g$ 是什么？

$$\Phi^T g = \Phi^T(\Phi f_\text{GT} + n) = \Phi^T\Phi f_\text{GT} + \Phi^T n$$

$\Phi^T\Phi f_\text{GT}$ 不是 $f_\text{GT}$——它是经过 mask + shift + compression 退化后再反投影的结果，包含严重的：

1. **Mask 调制伪影**：某些空间位置被 mask 遮挡，信息缺失
2. **Shift 错位**：不同波段在空间上偏移了 2×(band_index) 像素
3. **Compression 混叠**：28 个波段叠加成 1 层，光谱信息严重混合
4. **传感器噪声**：$\Phi^T n$ 放大后的噪声

**WPO 从这个"脏场"出发传播——物理方程是对的，但初始条件是错的。** 错误的初始条件经过波传播后会**扩散到全局**，甚至比没有全局传播还糟（局部错误变成全局错误）。

这解释了为什么源项注入（不断注入 $\Phi^T g$）反而变差——你在每个 stage 都重新注入"脏信号"，让 WPO 无法收敛到干净解。

### 2.3 ML 层为什么效果有限

ML 层（WSSA、FBA）确实提供了约 1 dB 的提升——这说明更精细的空间-光谱关联建模是有价值的。但问题在于：**这 1 dB 的代价是 4-5 倍的参数量**，而且随着训练推进，ML 层和纯 WPO 的差距在缩小（@10ep 差 0.68 dB，@60ep 差 1.16 dB，但 @300ep 估计差距缩回 0.5~0.8 dB）——说明长训练后 WPO 自身也能学到一部分关联。

ML 层被放在了"特征增强"的位置（和 WPO 并行残差），而不是"退化清理"的位置（WPO 之前）。这导致 ML 做的精细建模被输入中的退化伪影污染——效果打了折扣。

类比：在一张模糊照片上用更好的特征提取器，确实比差的提取器好一点，但远不如**先去模糊再提取**。

---

## 3. 回到问题本质：3D-WPO 到底做了什么

### 3.1 WPO 的核心能力

3D-WPO 解决的是**全局信息传播**问题：

$$\hat{u}(\boldsymbol{\omega}, t) = e^{-\alpha t/2}\left[\hat{u}_0\cos(\omega_d t) + \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\omega_d}\sin(\omega_d t)\right]$$

这个闭式解做了三件事：

1. **全局空间-光谱耦合**：3D FFT 让每个空间位置感知全局（$O(N\log N)$）
2. **频率均匀保留**：振荡式传播不会像热方程那样抑制高频
3. **物理结构先验**：色散关系 $\omega_d^2 = v_s^2|\boldsymbol{\omega}_{xy}|^2 + v_\lambda^2\omega_\lambda^2 - (\alpha/2)^2$ 编码了空间-光谱的传播速度差异

WPO 不擅长的：
- **局部纹理细节**：FFT 是全局操作，没有"局部自适应"能力
- **退化模式识别**：WPO 不知道 mask 长什么样、shift 了几个像素
- **噪声去除**：波传播不区分信号和噪声，传播多少信号就传播多少噪声

### 3.2 WPO 的物理角色重新定位

**WPO 不是 denoiser，是 propagator。**

在 deep unfolding 框架中，denoiser（prior network）的任务是"从被噪声/退化污染的中间估计中恢复干净信号"。这个任务需要：

1. 区分信号和噪声（WPO 做不到）
2. 利用图像先验（纹理、边缘、光谱相关性）（WPO 部分做到——光谱耦合）
3. 自适应不同退化程度（WPO 完全做不到——$\alpha, v_s, t$ 是全局标量）

把 WPO 单独当 denoiser 用，就像用波动方程去降噪——物理上不合理。波动方程描述的是"信号如何传播"，不是"如何从噪声中提取信号"。

**正确的角色分配**：让专门的退化处理模块先"净化"输入，然后把净化后的场交给 WPO 做全局传播。

---

## 4. 3D-WPO 对 HSI 重建的真实瓶颈

### 4.1 瓶颈 1：初始场质量差

从 $\Phi^T g$ 到初始场 $u_0$，目前只经过一个 `Conv2d(2C, C, 1)`。这个 1×1 卷积做的事情极其有限——它只能做通道混合，不能去除空间退化、不能补偿 shift 错位、不能分离混叠波段。

### 4.2 瓶颈 2：缺乏退化感知

WPO 的参数（$\alpha, v_s, v_\lambda, t$）在所有空间位置和所有退化程度下都相同。但 CASSI 的退化是**空间非均匀的**——mask 不同位置透射率不同，边界区域因 shift 导致信息密度低。

DPU 和 DAUHST 都意识到了这一点——它们用 degradation-aware 参数（从 mask 和 measurement 估计）来动态调整每个 stage 的行为。

### 4.3 瓶颈 3：stage 间缺乏退化残差学习

当前 GAP unfolding 的每个 stage：

```
GD step → WPO → 输出
```

每个 stage 的 WPO 独立工作，不知道"上一个 stage 修复了什么退化，还剩什么没修"。DPU 的 DPB（Degraded Prior Block）专门学习"退化残差"——每个 stage 估计当前的退化模式，明确告诉 prior network"你应该重点修什么"。

---

## 5. SOTA 方法的退化处理范式

### 5.1 三代退化处理

| 代际 | 方法 | 退化处理 | PSNR |
|------|------|---------|------|
| 0代 | GAP-Net (2023) | 无退化处理 | 33.26 |
| 1代 | DAUHST (NeurIPS 2022) | 从 mask+measurement 估计退化参数，控制迭代 | 38.36 |
| 1代 | RDLUF (CVPR 2023) | 学习 sensing matrix 的退化表示，混合空间-光谱先验 | 39.57 |
| **2代** | **DPU (CVPR 2024)** | **双先验分离：image prior + degradation prior，显式建模 mask+shift+compression** | **40.52** |
| 2代 | DERNN-LNLT (2024) | 退化估计网络（DEN）同时估计 sensing error 和 noise level | ~39.5 |
| 2代 | Phy-CoSF (2026) | DAN（Degradation-Aware Network）+ Nesterov 动量 + Fourier Mamba | ~40+ |

**趋势清晰**：退化处理越精细，性能越好。每一代的核心改进都不是"更大的 prior network"，而是"更准确的退化估计"。

### 5.2 DPU 的退化处理为什么有效

DPU 的 DPB（Degraded Prior Block）只有两个组件：

1. **退化 mask 构造**：把原始 mask 做 shift + compress + reverse，得到一个"包含全部退化信息的新 mask" $\Phi^*$
2. **退化权重估计**：一个 1×1 Conv + Sigmoid，从 $\Phi^*$ 和原始 $\Phi$ 的差异中学习逐像素退化权重

总参数量：约 **2 × C × C = 1568 参数**（C=28 时）。极其轻量。

但效果：**+1.13 dB**（从 38.49 到 39.62）。

**每增加 1K 参数带来 0.72 dB 提升**——和 WSSA 的 0.1 dB/M 相比，性价比是 7200 倍。

---

## 6. 新范式设计：净化-传播-精化

### 6.1 核心思路

```
2D measurement g + mask Φ
        │
        ▼
  ┌─────────────────┐
  │ 退化估计与初始场净化 │  ← 轻量 CNN，估计退化模式，生成"干净"初始场
  └────────┬────────┘
           │ 干净的 u₀, v₀
           ▼
  ┌─────────────────┐
  │    3D-WPO 传播    │  ← 从干净初始场出发，做物理全局传播
  └────────┬────────┘
           │ 传播后的波场
           ▼
  ┌─────────────────┐
  │   局部精化 + FFN   │  ← 轻量卷积修复 WPO 无法处理的局部细节
  └────────┬────────┘
           │
           ▼
       输出 f^{k+1}
```

**嵌入 deep unfolding**：每个 stage 重复上述三步。GD step 在"退化估计"之前，提供更新的 $f^k$。

### 6.2 与之前的关键区别

| 维度 | 之前（ML-WPO 堆砌） | 现在（净化-传播-精化） |
|------|-------------------|--------------------|
| ML 的角色 | 在 WPO 旁边做并行增强 | 在 WPO 之前做退化清理 |
| WPO 的输入 | 脏的（$\Phi^T g$ 直接进） | 净化后的（退化估计后的干净场）|
| 退化信息利用 | 无 | 显式估计 mask + shift + compression |
| 参数效率 | 差（3-4M 换 0.3 dB） | 好（DPB 风格，<0.01M 换 1+ dB）|
| WPO 的物理角色 | 被当 denoiser（不合理） | 被当 propagator（合理） |

---

## 7. 具体模块筛选与评估

### 7.1 退化估计模块（在 WPO 之前）

**最推荐：DPU 风格 Degraded Prior Block (DPB)**

理由：
- 已被 DPU 验证有效（+1.13 dB）
- 参数极少（<2K 参数）
- 物理上合理（显式利用 mask + shift + compression 的已知信息）
- 实现简单（一个 reverse 操作 + 1×1 Conv + Sigmoid）

```python
class DegradedPriorBlock(nn.Module):
    """参考 DPU/Model.py 的 DPB 设计
    
    从退化 mask Φ* 和原始 mask Φ 的差异中学习逐像素退化权重
    """
    def __init__(self, dim=28):
        super().__init__()
        # 退化权重估计
        self.weight_est = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),  # 学习 Φ* 和 Φ 的差异
            nn.LayerNorm([dim, 1, 1]),  # 或 InstanceNorm
            nn.Sigmoid()
        )
        # 特征变换
        self.feat_transform = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
    
    def forward(self, x, degraded_mask):
        """
        x: [B, C, H, W] 当前估计
        degraded_mask: [B, C, H, W] 经过 shift+compress+reverse 后的退化 mask
        """
        weight = self.weight_est(degraded_mask)   # [B, C, H, W] 退化权重
        feat = self.feat_transform(x)
        return weight * feat   # 退化加权
```

**可选增强：DAUHST 风格退化感知参数估计**

从 measurement 和 mask 估计当前退化程度的标量参数（noise level σ），传给 WPO 控制阻尼 α 和步长 ρ：

```python
class DegradationEstimator(nn.Module):
    """估计退化程度，控制 WPO 参数和 GD 步长"""
    def __init__(self, in_nc=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_nc, 32, 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),  # 输出 [noise_level, degradation_degree]
        )
    def forward(self, x, mask):
        return self.net(torch.cat([x, mask], dim=1) if False else x)
```

### 7.2 局部精化模块（在 WPO 之后）

WPO 做完全局传播后，局部细节可能不够精细。需要一个**轻量的局部精化**。

**不推荐**：WSSA（太重，3M+ 参数）、FBA（与 WPO 功能重叠）

**推荐**：简单的 DWConv + GELU + Conv1×1，即 FFN 的变体。这是 MST、DPU、SSR 里标准的"通道混合 + 局部细节增强"模块，参数量约 3×dim²（~2.4K）。

```python
class LocalRefinement(nn.Module):
    """WPO 之后的局部精化：DWConv 补充局部纹理"""
    def __init__(self, dim, expand=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * expand, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * expand, dim * expand, 3, 1, 1, groups=dim * expand, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * expand, dim, 1, bias=False),
        )
    def forward(self, x):
        return self.net(x)
```

### 7.3 你提到的论文的评估

| 论文 | 核心思路 | 适用性 | 推荐 |
|------|---------|-------|------|
| **HOGformer** (arXiv 2504.09377) | HOG 梯度引导退化感知注意力 | 中——退化感知是对的，但 HOG 是传统特征，不确定在 CASSI 上合适 | △ 关注思路但不直接搬 |
| **AIRPNet** (IEEE 2024) | 多退化分支 + 退化感知加权 | 低——针对多任务联合恢复，我们是单任务 | ✗ |
| **SCGN/FBGW** (arXiv 2603.18834) | 频带引导加权，根据频带内容动态增强/抑制 | **高**——和 WPO 的频域操作天然配合，而且不增加太多参数 | **✓ 值得参考** |
| **HyPyraMamba** (GitHub) | 多尺度 PCA 降维 + 双路 Mamba | 中——PCA 降维思路有用，双路 Mamba 不适合我们 | △ PCA 部分值得参考 |
| **MODA/CAFR** (arXiv 2512.09489) | 跨层光谱注意力 + 光谱引导自适应融合 | 低——目标检测任务，和重建差异大 | ✗ |

**SCGN 的 FBGW（频带引导加权）**最值得参考：它在频域中根据每个频带的内容动态生成权重——信号频带增强，噪声频带抑制。这和 WPO 的频域调制天然融合，而且不像 FBA 那样做冗余的小波分解。

### 7.4 FBGW 如何融入 WPO

不是在 WPO 外面加 FBGW，而是**修改 WPO 内部的频域调制**：

目前 WPO 频域调制：
$$\hat{u}_\text{out}(\omega) = K(\omega, t) \cdot \hat{u}_0(\omega)$$

加入频带引导加权：
$$\hat{u}_\text{out}(\omega) = W_\text{FBGW}(\omega) \cdot K(\omega, t) \cdot \hat{u}_0(\omega)$$

其中 $W_\text{FBGW}(\omega)$ 从频谱统计量（而非额外网络）计算：

$$W_\text{FBGW}(\omega) = \sigma\left(\frac{|\hat{u}_0(\omega)|^2 - \bar{\sigma}^2}{|\hat{u}_0(\omega)|^2 + \bar{\sigma}^2}\right)$$

$\bar{\sigma}^2$ 是估计的噪声功率谱（可由 DegradationEstimator 提供）。信号频带 $|\hat{u}_0| \gg \bar{\sigma}$ 时 $W \to 1$（保留），噪声频带 $|\hat{u}_0| \approx \bar{\sigma}$ 时 $W \to 0$（抑制）。

**额外参数：0。** 只需要噪声水平估计（已由 DegradationEstimator 提供）。

---

## 8. 降维与参数效率

### 8.1 PCA 降维的适用性

CAVE 数据集 28 波段，文献显示 HSI 的有效维度通常在 5-10。PCA 降到 8-12 维可保留 99%+ 能量。

**但 CASSI 重建中直接用 PCA 有问题**：我们需要重建完整 28 波段的 HSI，如果在网络中间做 PCA 降维，需要在输出时再升维回 28——这个升维过程可能引入伪影。

**更好的做法**：不在 feature 上做 PCA，而是在 **WPO 的频域操作中隐式降维**——只保留前 r 个主要的光谱模式做传播，其他模式直接跳过。这是之前讨论的 Low-Rank WPO 思路，但现在不是通过 SVD 截断（增加计算），而是通过 **1×1 Conv 投影到低维再投影回来**：

```python
# 在 WPO 之前：28 → r 维投影
u0_low = proj_down(u0)    # Conv2d(28, r, 1), r=8
# WPO 在 r 维空间传播（3D FFT 变小，计算量下降）
u_out_low = WPO3D_r(u0_low)  # dim=r 而非 dim=28
# 投影回 28 维
u_out = proj_up(u_out_low)   # Conv2d(r, 28, 1)
```

计算量从 $O(28 \cdot HW \log(28HW))$ 降到 $O(8 \cdot HW \log(8HW))$——**约降低 70%**。

### 8.2 与 DPU 的 MPMLP 对比

DPU 的 Multi-Pattern MLP 本质上也是一种"通道降维→处理→升维"操作（分组卷积 + shuffle）。我们的低维 WPO 投影是同样的思路，但用物理（光谱低秩性）来 justify。

---

## 9. 最终推荐架构

### 9.1 单 stage 内部（prior network）

```
输入 z（GD step 输出）+ mask Φ + degraded mask Φ*
    │
    ├── DegradedPriorBlock(z, Φ*)          → 退化权重 w
    │      (退化估计，~2K params)
    │
    ├── z_clean = z * w + z               → 净化后的初始场
    │      (退化加权 + 残差)
    │
    ├── [可选] proj_down(z_clean, 28→r)   → 低维投影
    │
    ├── WPO3D(z_clean, mask)              → 物理全局传播
    │      (已有模块，不改动)
    │
    ├── [可选] proj_up(wpo_out, r→28)     → 升维回来
    │
    ├── LocalRefinement(wpo_out)          → 局部细节精化
    │      (~2.4K params, DWConv+GELU+Conv)
    │
    └── 输出 f^{k+1} = z + wpo_out + local_out
           (残差连接)
```

### 9.2 完整 unfolding 流程

```
f^0 = init_conv(reverse(g), Φ)

For k = 0, ..., K-1:
    # GD step
    rho_k = ParaEstimator(f^k)
    z = f^k + rho_k * Φᵀ(g - Φf^k) / ΦΦᵀ
    
    # 退化估计
    Φ* = construct_degraded_mask(Φ)     # 一次性预计算
    w = DPB(z, Φ*)
    z_clean = z * w + z
    
    # 物理传播
    f_wave = WPO3D(z_clean, Φ)
    
    # 局部精化
    f_local = LocalRefine(f_wave)
    
    # 输出
    f^{k+1} = z + f_wave + f_local
    out_list.append(f^{k+1})
```

### 9.3 参数量估计

| 组件 | 参数量 | 说明 |
|------|-------|------|
| WPO3D（不变） | 0.79M | 已有 |
| ParaEstimator × K | ~0.02M×K | 已有 |
| DPB × K (shared) | ~0.002M | 新增，极轻量 |
| LocalRefine × K (shared) | ~0.003M | 新增，极轻量 |
| init_conv | ~0.002M | 已有 |
| **总计（shared, K=5）** | **~0.90M** | 几乎不增加 |
| **总计（non-shared, K=5）** | **~3.6M** | 主要是 5 份 WPO |

**对比 WSSA+alternating 的 3.67M → ~35.2 dB（@300ep 估计）**：新方案用 ~0.90M 参数预期达到 **39+ dB**（基于退化建模的 1+ dB 提升）。

### 9.4 论文叙事

> 我们观察到 3D-WPO 的性能瓶颈不在波传播算子本身，而在其输入——经过 mask、shift、compression 三重退化的初始场。我们提出"净化-传播-精化"三阶段 prior 设计：
>
> 1. **退化感知净化**（DPB）：从已知的 CASSI 物理退化模式中估计逐像素退化权重，生成干净初始场
> 2. **物理全局传播**（3D-WPO/KG）：在干净初始场上做波动方程闭式解传播
> 3. **局部细节精化**（DWConv FFN）：补充 WPO 无法处理的局部纹理
>
> 这个设计将 WPO 从"被迫做 denoiser"解放为"专注做 propagator"，让每个模块做自己最擅长的事。

---

## 10. 实验路线图

### 10.1 推荐顺序

**Phase 1（2 天）：端到端验证 DPB 的效果**

```
配置：WPO + DPB + LocalRefine, 端到端, 不含 unfolding
基线：纯 WPO 端到端 = 34.70 dB (@300ep)
目标：> 35.0 dB (@100ep)

关键：DPB 的退化 mask 构造（参考 DPU/Model.py 的 reverse 操作）
```

如果 @100ep 超过 35.0 → DPB 有效，进入 Phase 2

**Phase 2（3 天）：5-stage unfolding + DPB**

```
配置：[GD + DPB + WPO + LocalRefine] × 5 stage
基线：纯 WPO 5stg = 38.21 dB (@232ep)
目标：> 39.0 dB (@200ep)
```

**Phase 3（1 周）：完整消融 + KG 版本**

```
消融：
  - WPO only (baseline)                → 38.21
  - WPO + DPB                          → 预期 39.0+
  - WPO + DPB + LocalRefine            → 预期 39.2+
  - WPO + DPB + LR + Low-Rank(r=8)     → 预期 39.2+, 更快
  - KG + DPB + LR                      → 预期 39.2+, SAM 最低
```

### 10.2 不再做的事

- ✗ ML 层堆砌（WSSA、FBA、Wave-Aware）——已证明性价比极差
- ✗ 源项注入——已证明有害
- ✗ 色散介质——理论优美但实际收益未知，优先级降低
- ✗ 小波分解——与 WPO 功能重叠

### 10.3 论文时间线

Phase 1-2 验证成功后（约 1 周），开始写论文。论文的核心表格：

| 模型 | PSNR | SSIM | SAM | Params | FLOPs |
|------|------|------|-----|--------|-------|
| MST | 35.18 | 0.948 | — | 2.03M | 28.15G |
| DAUHST-9stg | 38.36 | 0.967 | — | 6.15M | 79.5G |
| DPU-5stg | 39.62 | 0.973 | — | 1.59M | 27.41G |
| SSR-S | 39.19 | 0.971 | — | 1.73M | 26.37G |
| **Ours-5stg** | **~39.2** | **~0.97** | **~0.08** | **~0.90M** | **~18G** |

即使 PSNR 不到 SOTA，**参数量和 FLOPs 远低于所有竞品**——这是物理先验+精准退化估计的优势。

