# WPO-Propagator 新框架：退化估计、WPO 增强、降维、展开框架的逐项决策

> **原则**：每个决策都基于已有实验数据或已发表论文的消融实验。不确定的标注"待验证"，不合适的直接否决。不讨好。

---

## 目录

1. [决策总览](#1-决策总览)
2. [退化估计：不能照搬 DPU，但也不能不做](#2-退化估计不能照搬-dpu但也不能不做)
3. [WPO 内部增强：FBGW 频带引导加权](#3-wpo-内部增强fbgw-频带引导加权)
4. [降维：是否需要，如何做](#4-降维是否需要如何做)
5. [展开框架：GAP vs A-HQS vs ADMM](#5-展开框架gap-vs-a-hqs-vs-admm)
6. [完整架构方案](#6-完整架构方案)
7. [参数量预算与对比](#7-参数量预算与对比)
8. [实验计划](#8-实验计划)

---

## 1. 决策总览

| 决策项 | 结论 | 信心 | 依据 |
|-------|------|------|------|
| 退化估计 | **做，但不照搬 DPU 的 DPB** | 高 | DPU +1.13 dB, DERNN-LNLT 验证有效 |
| FBGW 融入 WPO | **做，零参数增加** | 中 | SCGN 论文验证有效，与 WPO 天然融合 |
| PCA / 降维 | **不做显式 PCA，用 1×1 Conv 隐式降维** | 中高 | 28 波段已经不算高维，显式 PCA 破坏端到端训练 |
| 展开框架 | **从 GAP 升级到 A-HQS（加动量）** | 高 | Phy-CoSF/CA²UN 已验证，理论有支撑 |
| ML 层堆砌 | **不做** | 高 | 实验已证明性价比差 |
| 色散介质 | **不做** | 中高 | 实验已证明无明显收益 |
| 源项注入 | **不做** | 高 | 实验已证明有害 |

---

## 2. 退化估计：不能照搬 DPU，但也不能不做

### 2.1 为什么不能照搬 DPU 的 DPB

DPU 的 DPB 做了什么：

```
1. 把 mask 做 shift → compress → reverse，得到退化 mask Φ*
2. 一个 1×1 Conv 学习 Φ 和 Φ* 的差异
3. Sigmoid 输出退化权重
4. 权重逐元素乘到特征上
```

这个设计**只估计了退化的空间分布**（哪些位置退化严重），没有估计：

- **退化的程度**（noise level σ，影响 denoiser 的强度）
- **退化的类型**（sensing error，即理想 Φ 和真实退化过程之间的差异）
- **退化的光谱依赖性**（不同波段因 shift 导致的信息密度不同）

DPU 的 DPB 之所以够用，是因为 DPU 的 prior network（Focused Transformer）本身很强——它依靠 Transformer 的大容量来隐式学习剩余的退化模式。我们的 WPO 容量远小于 Transformer，**无法隐式补偿 DPB 遗漏的退化信息**。

### 2.2 DERNN-LNLT 的退化估计更适合我们

DERNN-LNLT (2024) 提出了 **DEN（Degradation Estimation Network）**，同时估计两个东西：

1. **退化矩阵 $\Delta\Phi$**：sensing matrix 的误差修正。$\Phi_\text{real} = \Phi_\text{ideal} + \Delta\Phi$

   $$x^{k+1} = (\Phi + \Delta\Phi)^T \frac{y - (\Phi + \Delta\Phi) z^k}{\text{denom}} \tag{2.1}$$

   用 $\Phi + \Delta\Phi$ 替代 $\Phi$ 做 GD step，修正因 sensing error 导致的数据保真偏差。

2. **噪声水平 $\sigma$**：作为条件信息传给 prior network，让 denoiser 根据当前噪声水平动态调整去噪强度。

DEN 的实现：以 $\Phi$ 为参考，通过残差学习估计 $\Delta\Phi$。

$$\Delta\Phi = \text{Conv}_{1\times1}(\Phi) \tag{2.2}$$
$$\sigma = \text{MLP}(\text{GlobalAvgPool}(\Phi)) \tag{2.3}$$

参数量：约 $2 \times C^2 + C \times 32 + 32 = 2.5K$（C=28 时）。

### 2.3 我们的退化估计方案

结合 DPU 和 DERNN-LNLT 的优点，设计一个**三合一退化估计模块**：

```python
class DegradationEstimation(nn.Module):
    """三合一退化估计：退化权重 + sensing error + noise level
    
    输入：
      - f: [B, C, H, W] 当前迭代估计
      - Phi: [B, C, H, W] spatial mask
      - Phi_star: [B, C, H, W] degraded mask (shift+compress+reverse)
    
    输出：
      - delta_Phi: [B, C, H, W] sensing error (修正 GD step)
      - deg_weight: [B, C, H, W] 退化权重 (净化初始场)
      - sigma: [B, 1, 1, 1] 噪声水平 (传给 WPO 控制阻尼)
    """
    def __init__(self, dim=28, hidden=32):
        super().__init__()
        # 1. Sensing error 估计 (参考 DERNN-LNLT)
        self.delta_phi = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        
        # 2. 退化空间权重 (参考 DPU)
        self.deg_weight = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=False),  # cat(Phi, Phi*)
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )
        
        # 3. 噪声水平估计 (从 f 和 Phi 估计)
        self.sigma_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # σ > 0
        )
    
    def forward(self, f, Phi, Phi_star):
        # Sensing error
        delta_Phi = self.delta_phi(Phi)  # [B, C, H, W]
        
        # 退化空间权重
        deg_weight = self.deg_weight(torch.cat([Phi, Phi_star], dim=1))
        
        # 噪声水平
        sigma = self.sigma_est(f).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        return delta_Phi, deg_weight, sigma
```

**参数量**：约 $2C^2 + 2 \times 2C \times 32 + 32 + C \times 32 + 32 \approx 5.5K$（C=28 时）。极轻量。

### 2.4 三个输出怎么用

| 输出 | 用在哪里 | 怎么用 |
|------|---------|-------|
| $\Delta\Phi$ | **GD step** | $\Phi_\text{eff} = \Phi + \Delta\Phi$，替代原始 $\Phi$ 做数据保真 |
| deg_weight | **WPO 输入** | $u_0 = \text{deg\_weight} \cdot z + z$（净化后的初始场） |
| $\sigma$ | **WPO 参数** | 动态调整 WPO 的阻尼 $\alpha_\text{eff} = \alpha + \lambda\sigma$（噪声大→阻尼大→传播保守） |

**$\sigma$ 对 WPO 的意义**：噪声水平高时，WPO 应该"保守传播"（高阻尼，抑制高频），避免把噪声扩散到全局；噪声水平低时，WPO 应该"大胆传播"（低阻尼，保留高频细节）。这是**退化感知的物理传播**——物理参数由退化估计动态控制。

---

## 3. WPO 内部增强：FBGW 频带引导加权

### 3.1 SCGN 的 FBGW 做了什么

SCGN（arXiv 2603.18834）的 Frequency Band-Guided Weighting：

```
1. 把特征做 FFT
2. 把频谱分成若干频带
3. 计算每个频带的能量统计量
4. 结合频带位置信息生成权重
5. 权重作用于频域特征
6. iFFT 回空间域
```

核心思想：不同频带中信号和噪声的分布不同，动态增强信号频带、抑制噪声频带。

### 3.2 融入 WPO 的方案

WPO 本身就在频域操作——**不需要额外做 FFT/iFFT**，只需在 WPO 的频域调制步骤中加入一个**自适应频带权重**。

**当前 WPO 频域调制**：

$$\hat{u}_\text{out}(\boldsymbol{\omega}) = e^{-\alpha t/2}\left[\hat{u}_0 \cos(\omega_d t) + \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\omega_d}\sin(\omega_d t)\right]$$

**加入 FBGW**：

$$\hat{u}_\text{out}(\boldsymbol{\omega}) = W(\boldsymbol{\omega}) \cdot e^{-\alpha t/2}\left[\cdots\right]$$

其中 $W(\boldsymbol{\omega})$ 是频带引导权重。

### 3.3 $W(\boldsymbol{\omega})$ 的计算

有两种设计：

**方案 A：基于信噪比的自适应权重（零参数）**

$$W(\boldsymbol{\omega}) = \sigma_\text{gate}\left(\frac{|\hat{u}_0(\boldsymbol{\omega})|^2 - \sigma^2}{|\hat{u}_0(\boldsymbol{\omega})|^2 + \sigma^2 + \epsilon}\right) \tag{3.1}$$

- $|\hat{u}_0(\boldsymbol{\omega})|^2$ 是初始场在该频率的功率
- $\sigma^2$ 是从 DegradationEstimation 得到的噪声功率（可直接用 §2 的输出）
- 信号频带 $|\hat{u}_0| \gg \sigma$：$W \to 1$（保留）
- 噪声频带 $|\hat{u}_0| \approx \sigma$：$W \to 0$（抑制）
- $\sigma_\text{gate}$ 是 sigmoid，让权重在 [0, 1]

**额外参数：0。** 只利用了已有的 $\sigma$ 和 $\hat{u}_0$。

**方案 B：可学习的频带权重（少量参数）**

把频率空间分成 $K$ 个频带（按 $|\boldsymbol{\omega}|$ 的大小分），每个频带学习一个缩放因子：

$$W(\boldsymbol{\omega}) = w_{\text{band}(|\boldsymbol{\omega}|)} \in \mathbb{R}^K, \quad K = 4 \text{ 或 } 8$$

**额外参数：K 个标量。** 4-8 个 float32。

### 3.4 推荐

**先用方案 A（零参数），验证是否有效。** 如果有效再尝试方案 B。

方案 A 的物理意义清晰：它是经典信号处理中 **Wiener 滤波器**的简化版——根据信噪比自适应选择保留或抑制每个频率分量。把 Wiener 滤波嵌入 WPO 的频域调制中，是**物理先验（波传播）+ 统计先验（信噪比估计）**的融合，不是缝合。

### 3.5 与 FBA（小波方案）的区别

| 维度 | FBA（已否决） | FBGW 融入 WPO |
|------|------------|--------------|
| 额外 FFT | 需要（小波分解 + 重建） | **不需要**（复用 WPO 已有的 FFT） |
| 额外参数 | 4.47M（含 QKV 投影等） | **0**（方案 A）或 4-8（方案 B） |
| 与 WPO 的关系 | 外挂（Block 级别并行） | **内嵌**（在 WPO 频域调制步骤中） |
| 计算量增加 | ~90%（翻倍训练时间） | **~0%**（只多一次逐元素乘法） |
| 功能重叠 | 严重（小波≈FFT） | **无**（频带加权 ≠ 波传播调制） |

---

## 4. 降维：是否需要，如何做

### 4.1 分析

CAVE 28 波段的有效维度：文献一致报告 HSI 数据用 PCA 降到 5-10 维可保留 99%+ 能量。但 28 波段本身**已经不算高维**——对比遥感 HSI 动辄 200+ 波段。

在 CASSI unfolding 文献中：

| 方法 | 是否降维 | PSNR |
|------|---------|------|
| MST | 否 | 35.18 |
| DPU | 否 | 40.52 |
| SSR | 否 | 40.69 |
| DERNN-LNLT | 否（RNN 共享参数代替降维） | ~39.5 |
| RDLUF | 否 | 39.57 |

**没有一个 CASSI SOTA 用显式 PCA 降维。** 它们通过其他方式（共享权重、低秩 attention、分组卷积）隐式控制参数量。

### 4.2 结论

**不做显式 PCA 降维。** 理由：

1. 28 维不构成维度灾难——WPO 的 3D FFT 对 28 通道的开销可控
2. 显式 PCA 需要在训练前做特征分析，破坏端到端训练
3. 没有 CASSI SOTA 方法用 PCA——说明这不是瓶颈
4. 如果需要参数效率，用**共享权重**（DERNN-LNLT 已验证，参数降为 1/K）比 PCA 更有效

**但可以用 1×1 Conv 做隐式低秩投影（可选）**：在 WPO 前加 `Conv2d(28, r, 1)` 投影到 r=12 维，WPO 在低维空间传播，再 `Conv2d(r, 28, 1)` 投影回来。参数增加约 $2 \times 28 \times 12 = 672$，计算量下降约 57%。**作为消融实验选项，不作为默认配置。**

---

## 5. 展开框架：GAP vs A-HQS vs ADMM

### 5.1 三种框架的数学形式

**GAP（当前使用）**：

$$f^{k+1} = \text{Prior}\left(f^k + \rho_k \Phi^T\frac{g - \Phi f^k}{\Phi\Phi^T}\right) \tag{5.1}$$

一阶，无动量，无对偶变量。

**A-HQS（推荐升级）**：

$$x^{k+1} = (\Phi^T\Phi + \mu I)^{-1}(\Phi^T y + \mu \hat{z}^k) \tag{5.2}$$
$$z^{k+1} = \text{Prior}(x^{k+1}, \sigma_k) \tag{5.3}$$
$$\hat{z}^{k+1} = z^{k+1} + \beta_k(z^{k+1} - z^k) \tag{5.4}$$

二阶（有 Nesterov 动量 $\beta_k$），数据子问题有闭式解。

**ADMM（DPU 使用）**：

$$r^{k+1} = \text{DPB}(\ldots) \tag{5.5}$$
$$z^{k+1} = \text{IPB}(\ldots) \tag{5.6}$$
$$f^{k+1} = \text{GD}(z^{k+1}, r^{k+1}, y^k) \tag{5.7}$$
$$y^{k+1} = y^k + \mu^{k+1}(f^{k+1} - z^{k+1} + r^{k+1}) \tag{5.8}$$

二阶（Lagrange 乘子 $y$ = Hamilton 动量），双 prior。

### 5.2 决策分析

| 框架 | 收敛速率 | 实现复杂度 | 已有验证 |
|------|---------|----------|---------|
| GAP | $O(1/K)$ | 低 | 我们的 38.21 dB |
| **A-HQS** | $O(1/K^2)$ | **中** | Phy-CoSF、CA²UN 已验证 |
| ADMM | $O(1/K^2)$ | 高 | DPU 40.52 dB |

### 5.3 推荐：升级到 A-HQS

理由：

1. **动量已被多篇论文验证有效**——Phy-CoSF (2026)、CA²UN (IET 2025)、GA-HQS (arXiv 2023) 都在 CASSI/MRI 展开中使用了 A-HQS + Nesterov 动量，获得了一致的提升。

2. **改动极小**——只在 GAP 的基础上加一行动量外推：

   ```python
   # GAP (当前):
   f = Prior(z)
   
   # A-HQS (升级):
   f = Prior(z)
   f_accel = f + beta[k] * (f - f_prev)  # 加一行
   f_prev = f
   f = f_accel
   ```

3. **不需要引入 Lagrange 乘子**——比 ADMM 简单得多，但理论收敛速率相同（$O(1/K^2)$）。

4. **$\beta_k$ 的选择**：CA²UN 和 Phy-CoSF 都让 $\beta_k$ 可学习（每 stage 一个标量），参数增加 K 个 float32。

### 5.4 GD step 的改进

A-HQS 的数据子问题 (5.2) 有一个**精确闭式解**（比 GAP 的梯度下降更准确）：

$$x^{k+1} = \hat{z}^k + \Phi^T\frac{y - \Phi\hat{z}^k}{\mu + \Phi\Phi^T} \tag{5.9}$$

和 GAP 的公式形式相同，但输入是**动量外推后的** $\hat{z}^k$，不是原始 $z^k$。

结合 §2 的 sensing error 修正：

$$x^{k+1} = \hat{z}^k + (\Phi + \Delta\Phi)^T\frac{y - (\Phi + \Delta\Phi)\hat{z}^k}{\mu + (\Phi + \Delta\Phi)(\Phi + \Delta\Phi)^T} \tag{5.10}$$

### 5.5 与 Least Action 的关系

A-HQS + Nesterov 正是我们在 Least Action Analysis 中从 Euler-Lagrange 方程推导出的**二阶动力学**。区别是：

- CA²UN/Phy-CoSF 从"优化加速"角度引入动量（工程动机）
- 我们从"最小作用量原理"引入动量（物理动机）

**论文 1 不需要讲这个理论**——直接用 A-HQS 即可，引用 CA²UN 作为方法来源。论文 2 再展开物理推导。

---

## 6. 完整架构方案

### 6.1 单 stage 流程

```
输入: f^k (上一 stage 输出), g (测量值), Φ (mask), Φ* (退化 mask, 预计算)

1. 退化估计
   ΔΦ, w_deg, σ = DegradationEstimation(f^k, Φ, Φ*)
   
2. 动量外推 (A-HQS)
   ẑ = f^k + β_k * (f^k - f^{k-1})      # Nesterov 动量
   
3. 修正 GD step (数据保真)
   Φ_eff = Φ + ΔΦ                        # sensing error 修正
   x = ẑ + Φ_eff^T (g - Φ_eff ẑ) / (μ + Φ_eff Φ_eff^T)
   
4. 初始场净化
   u0 = x * w_deg + x                    # 退化加权 + 残差
   
5. WPO 传播 (物理核心)
   α_eff = α + λ * σ                     # 噪声感知阻尼
   [可选] u0_fft = FFT(u0)
   [可选] u0_fft = FBGW(u0_fft, σ)       # 频带引导加权
   f_wave = WPO3D(u0, mask, α_eff)       # 波传播 (内部用 α_eff)
   
6. 局部精化
   f_local = DWConv_FFN(f_wave)           # 轻量局部增强
   
7. 输出
   f^{k+1} = x + f_wave + f_local        # 三路残差

输出: f^{k+1}, 存入 out_list
```

### 6.2 与之前方案的关键区别

| 变化点 | 之前 (stage2) | 现在 |
|-------|-------------|------|
| 展开框架 | GAP（一阶） | **A-HQS（二阶，+动量）** |
| 退化估计 | 无 | **三合一（ΔΦ + w_deg + σ）** |
| GD step | $\Phi$ 固定 | **$\Phi + \Delta\Phi$ 修正** |
| WPO 阻尼 | 固定 $\alpha$ | **$\alpha + \lambda\sigma$（噪声感知）** |
| 频域增强 | 无 | **FBGW（零参数或 K 参数）** |
| 局部精化 | 无 | **DWConv FFN（~2.4K 参数）** |
| ML 层堆砌 | WSSA/FBA | **删除** |

---

## 7. 参数量预算与对比

### 7.1 单 stage 参数量

| 组件 | 参数量 | 来源 |
|------|-------|------|
| WPO3D U-Net | 0.79M | 已有，不改 |
| DegradationEstimation | ~5.5K | 新增 |
| LocalRefinement (DWConv FFN) | ~4.7K | 新增 |
| ParaEstimator ($\rho_k$) | ~10K | 已有 |
| $\beta_k$ (动量系数) | 1 scalar | 新增 |
| FBGW (方案 A) | 0 | 新增 |
| **单 stage 总计** | **~0.81M** | |

### 7.2 不同配置

| 配置 | 总参数量 | 对比 |
|------|---------|------|
| 5-stage shared | ~0.87M | vs 纯 WPO 5stg 0.85M（+2%） |
| 5-stage non-shared | ~4.1M | vs DPU-5stg 1.59M（2.6×，但含完整 WPO U-Net） |
| 9-stage shared | ~0.90M | vs DPU-9stg 2.85M（0.3×） |
| **9-stage non-shared** | **~7.3M** | vs SSR-L 5.18M（1.4×） |

**可以大胆一点**：non-shared 5-stage 的 4.1M 参数完全合理——DPU-5stg 用 1.59M 已经是 SOTA，我们 4.1M 有更多余量给退化估计和 WPO 各自的 stage-specific 调整。

### 7.3 推荐配置

**主推**：5-stage non-shared，预计 ~4.1M 参数。

**消融**：5-stage shared，预计 ~0.87M 参数。如果 shared 就能达到好效果，说明"退化估计 + FBGW + 动量"的改进是框架层面的，不依赖参数量。

---

## 8. 实验计划

### 8.1 端到端快速验证（不含 unfolding）

**目的**：验证退化估计 + FBGW 对 WPO 的提升。

| 实验 | 配置 | baseline | 预期 | 验证什么 |
|------|------|----------|------|---------|
| E1 | WPO only (现有) | — | 34.70 @300ep | baseline |
| E2 | WPO + DegEst (只 w_deg) | +5.5K | > 35.0 | 退化净化是否有效 |
| E3 | WPO + DegEst + FBGW | +5.5K | > 35.2 | 频带加权是否有效 |
| E4 | WPO + DegEst + FBGW + LocalRefine | +10K | > 35.3 | 局部精化是否有效 |

**各跑 10 epoch**（根据之前分析，10 epoch 足够判断趋势）。

### 8.2 Unfolding 验证

| 实验 | 配置 | baseline | 预期 |
|------|------|----------|------|
| U1 | GAP 5stg (现有) | — | 38.21 @232ep |
| U2 | A-HQS 5stg (加动量) | +5 scalars | > 38.5 |
| U3 | A-HQS + DegEst 5stg | +5.5K×5 | > 39.0 |
| U4 | A-HQS + DegEst + FBGW + LR 5stg | +10K×5 | > 39.2 |
| U5 | U4 + sensing error 修正 (ΔΦ) | +2.5K×5 | > 39.3 |
| U6 | U4 non-shared | ~4.1M | > 39.5 |

**U2 是最关键的实验**——如果加动量就能从 38.21 提到 38.5+，说明展开框架的改进是实质性的。这只需要改 3 行代码。

### 8.3 KG 版本

在最佳配置上加 KG（$k^2(\lambda)$ 注入色散关系），预期 SAM 显著下降（光谱保真提升），PSNR 持平或略有波动。

### 8.4 消融表设计

论文最终的消融表：

```
| 配置                            | PSNR  | SSIM  | SAM   | Params |
|--------------------------------|-------|-------|-------|--------|
| WPO (e2e baseline)             | 34.70 | 0.943 | 0.134 | 0.79M  |
| WPO + GAP 5stg                 | 38.21 | 0.970 | 0.079 | 0.85M  |
| WPO + A-HQS 5stg (动量)        | ~38.6 | ~0.972| ~0.077| 0.85M  |
| + DegEst                       | ~39.0 | ~0.973| ~0.075| 0.88M  |
| + FBGW                         | ~39.2 | ~0.974| ~0.074| 0.88M  |
| + LocalRefine                  | ~39.3 | ~0.974| ~0.073| 0.90M  |
| + ΔΦ sensing error             | ~39.4 | ~0.975| ~0.072| 0.90M  |
| + non-shared                   | ~39.7 | ~0.976| ~0.070| 4.1M   |
| KG 版本 (non-shared)            | ~39.6 | ~0.975| ~0.065| 4.1M   |
```

KG 的 PSNR 可能略低于 WPO（KG 约束更强，灵活性降低），但 SAM 预期显著更低——这是论文核心卖点。

### 8.5 不确定性标注

以下结论基于理论推导和其他论文的消融数据推算，**需要实验验证**：

| 预期 | 信心 | 依据 |
|------|------|------|
| A-HQS 动量提升 0.3-0.5 dB | **中高** | Phy-CoSF/CA²UN 报告了类似提升 |
| 退化估计提升 0.5-1.0 dB | **中** | DPU 报告 +1.13 dB，但我们的 prior 更弱 |
| FBGW 提升 0.1-0.3 dB | **中低** | SCGN 在电镜去噪上有效，CASSI 待验证 |
| Sensing error 修正提升 0.1-0.2 dB | **中** | DERNN-LNLT 报告有效 |
| KG SAM 改善 0.01+ | **中高** | 单 stage 实验已验证 SAM 0.1328 vs 0.1343 |

**如果某个组件在实验中无效（提升 < 0.05 dB），直接删除，不要强行保留。**

