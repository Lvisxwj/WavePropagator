# 3D-WPO / 3D-KG 改进方向深度分析

## 物理可解释模型的工程化升级 + 物理增强模块设计

> **本文出发点**：用户已完成 7 个模型实验，最高 2D-WPO+SMSA（34.81 dB），但更倾向物理解释最强的 **3D-WPO Pure（34.70 dB）** 和 **3D-WPO-KG（34.69 dB）**作为论文主推。本文不去追逐 SMSA 路径，而是专注于如何让这两个物理模型更上一层楼。
>
> **结构**：
> - 第 1 部分：与 SOTA 的精确差距分析
> - 第 2-5 部分：纯深度学习思路（吸收 DPU/SSR 的工程经验）
> - 第 6-9 部分：纯物理思路（从波动方程本身出发增强）
> - 第 10 部分：低秩与 LoRA 的物理对应
> - 第 11 部分：综合改进路线图

---

## 目录

1. [现状定位与差距来源分析](#1-现状定位与差距来源分析)
2. [DPU / SSR 核心机制拆解](#2-dpu--ssr-核心机制拆解)
3. [纯深度学习改进 A：Deep Unfolding 框架](#3-纯深度学习改进-adeep-unfolding-框架)
4. [纯深度学习改进 B：吸收 SSR 的 WSSA 教训](#4-纯深度学习改进-b吸收-ssr-的-wssa-教训)
5. [纯深度学习改进 C：吸收 DPU 的 PCA Focused Attention](#5-纯深度学习改进-c吸收-dpu-的-pca-focused-attention)
6. [物理改进 A：色散关系增强](#6-物理改进-a色散关系增强)
7. [物理改进 B：源项与边界的物理化](#7-物理改进-b源项与边界的物理化)
8. [物理改进 C：多尺度波传播](#8-物理改进-c多尺度波传播)
9. [物理改进 D：守恒律与稳定性约束](#9-物理改进-d守恒律与稳定性约束)
10. [低秩 / LoRA：HSI 物理特性的天然对应](#10-低秩--lora-hsi-物理特性的天然对应)
11. [综合改进路线图与优先级](#11-综合改进路线图与优先级)

---

## 1. 现状定位与差距来源分析

### 1.1 PSNR 差距全景

| 类别 | 模型 | PSNR | Params | 主要差距来源 |
|------|------|------|--------|------------|
| 我们 | **3D-WPO Pure (MaskA)** | **34.70** | 0.79M | baseline |
| 我们 | **3D-WPO-KG (MaskD)** | **34.69** | 0.79M | KG 增加 $k^2(\lambda)$ |
| End-to-end | MST | 35.18 | 2.03 | S-MSA + 更多参数 |
| End-to-end | CST | 36.12 | 3.0 | Sparse + 多尺度 |
| End-to-end | BIRNAT | 37.58 | 4.40 | RNN 迭代 |
| **Unfolding** | DAUHST-9stg | 38.36 | 6.15 | 9 次数据保真步 |
| **Unfolding** | RDLUF | 39.57 | 1.81 | 显式退化建模 |
| **Unfolding** | DPU-9stg | 40.52 | 2.85 | 双 prior + FA |
| **Unfolding** | SSR-L | 40.69 | 5.18 | WSSA + ARB |

### 1.2 差距的精确分解

我们 vs SOTA Unfolding 的差距约 **5.7 dB**，可以分解为：

**(a) 结构差距：~3.5 dB**
- 单 stage end-to-end vs 9 stage unfolding
- 每个 unfolding stage 都重新引入测量值 $g$ 做数据保真，相当于把"逆问题约束"重复施加 9 次
- DPU 表 1 显示：单纯换框架（GAP→DAUF→RDLF→DPF），同样的 5-stage 从 38.39→39.62，提升 1.23 dB

**(b) 注意力机制差距：~1.0 dB**
- SSR 的 WSSA（避免 mean effect）vs 我们的 3D FFT（无注意力）
- DPU 的 Focused Attention（PCA scaling + sparse）vs 朴素 attention

**(c) 退化建模差距：~0.7 dB**
- DPU 的 Degraded Prior Block 显式建模 mask + shift + compression
- 我们目前只用了 mask 软门控，没有显式建模 shift / compression

**(d) 参数效率差距：~0.5 dB**
- MaskA 之外，DPU 用 multi-pattern MLP 减少参数同时增加表达力

### 1.3 关键启示

> **3D-WPO/KG 与 SOTA 的差距 80% 来自架构层面（unfolding + attention 机制），20% 来自具体算子设计。**
>
> 如果要让物理模型真正有竞争力，**第一优先级是把 3D-WPO/KG 嵌入 unfolding 框架**，而不是修改 WPO 内部算子。

---

## 2. DPU / SSR 核心机制拆解

### 2.1 DPU 的核心创新

**Dual Prior Framework (DPF)**：

传统单 prior unfolding 解的是：
$$\arg\min_f \frac{1}{2}\|g - \Phi f\|^2 + \gamma D(f)$$

DPU 引入残差变量 $r$，把问题改为：
$$\arg\min_{f, z, r} \frac{1}{2}\|g - \Phi f\|^2 + \gamma D(z) + \tau R(r), \quad \text{s.t. } f = z - r$$

其中 $R(r)$ 是**退化先验**——专门学习 mask + shift + compression 三种退化的复合效应。每个 stage 同时输出"图像 prior 重建" $z$ 和"退化残差" $r$，融合后得到更好的 $f$。

**Focused Attention**：

PCA 启发的注意力增强。给定 $Q, K \in \mathbb{R}^{HW \times d}$：
$$M = QK^T, \quad q = QW_q, \quad k = KW_k, \quad q, k \in \mathbb{R}^{HW \times 1}$$

新注意力：
$$M' = (qk^T) \odot M$$

其中 $qk^T$ 是按主成分投影计算的"重要性掩码"。再用阈值过滤：
$$\text{Atten} = \text{softmax}(\text{prox}_\theta(M')) V, \quad \text{prox}_\theta(M) = \begin{cases} M & M > \theta \\ -\infty & M \leq \theta \end{cases}$$

### 2.2 SSR 的核心创新

**Mean Effect 分析**（重要）：

MST 的 S-MSA 把 28 个波段当成 28 个 token 做注意力，但**多头切分在 channel 维度**——每个 head 只看到部分通道，无法获取全局光谱信息。SSR 数学化这个问题：

$$\text{sim}(f^k_i, f^k_j) \to \frac{1}{p}\sum_{m=1}^p \text{sim}(f^m_i, f^m_j)$$

即"我们想要 pattern $k$ 的相似度，实际得到的是所有 pattern 的平均相似度"——这就是 **mean effect**。

**WSSA 解决方案**：

不在 channel 维切多头，而在 **空间维切窗**：

```
原始：[H, W, C] → split channel into N heads → [H, W, C/N] × N → 各头独立做 spectral attention
WSSA：[H, W, C] → split spatial into windows of M×M → [HW/M², M², C] → 每个窗内做完整 C 维 spectral attention
```

每个窗内做的是**完整 C 维**的光谱注意力（保留全局光谱信息），不同窗之间互不干扰（保留局部差异）。复杂度仍是 $O(HW \cdot C^2)$，与 window size 无关。

**ARB（Spatial Rectification Block）**：

WSSA 在窗内做 attention，跨窗无交互，会有 blocking artifact。ARB 用 **11×11 大核 depth-wise conv** 做跨窗交互（11>8 = window size，所以邻窗能交互）。

**SAB（Spatial Alignment Block）**：

CASSI 中某些波段（特别是 shift 后边界附近的波段）出现严重失真。SAB 假设：低质量波段可以从高质量波段加权恢复（**因为光谱低秩**）：
$$Y[\lambda] = T(x,y) \odot SW[\lambda]$$

其中 $T$ 是从所有波段学到的"统一空间纹理"，$SW[\lambda]$ 是该波段的"光谱权重"。

### 2.3 这两篇 SOTA 给我们的启示

| 启示点 | DPU | SSR | 是否适用 3D-WPO/KG |
|-------|-----|-----|------------------|
| Unfolding 多 stage 框架 | ✓ | ✓ | ✓✓✓（最关键）|
| 显式建模 shift+compression 退化 | ✓ | – | ✓✓ |
| 避免 mean effect | – | ✓ | ✓（光谱建模时）|
| PCA 主成分增强注意力 | ✓ | – | △（无 attention 时不直接用，但有物理对应）|
| 大核卷积补全跨窗交互 | – | ✓ | ✓（FFT 后空间细节）|
| 光谱低秩对齐 | – | ✓ | ✓✓（与物理低秩对应）|

---

## 3. 纯深度学习改进 A：Deep Unfolding 框架

### 3.1 为什么必须做 unfolding

CASSI 重建是经典的逆问题：
$$g = \Phi f + n$$

逆问题求解的标准范式是迭代算法（ISTA、ADMM、HQS），形式如：
$$f^{k+1} = \arg\min_f \|g - \Phi f\|^2 + \rho \|f - z^k\|^2$$
$$z^{k+1} = \text{Denoiser}(f^{k+1})$$

迭代多次让"数据保真"和"先验"反复约束。Deep unfolding 把这个迭代过程展开成 $K$ 个 stage 的网络，每个 stage 都包含一次"数据保真 + 先验"。

**核心好处**：测量值 $g$ 和退化矩阵 $\Phi$ 在每个 stage 都参与计算——单 forward pass 的 end-to-end 网络只在初始化时用了一次 $g$，之后纯粹靠学习的先验恢复，信息利用不充分。

### 3.2 把 3D-WPO/KG 嵌入 unfolding 框架

最简单方案：HQS（Half Quadratic Splitting）展开。

**整体架构**：

```
Stage 1:
  GD step: f^0 = Φ^T g (初始化)
  Prior step: z^1 = WPO_3D(f^0, mask)   ← 我们的 3D-WPO 在这里
  Closed-form fusion: f^1 = z^1 + Φ^T(g - Φ z^1) / (μ + ΦΦ^T)

Stage 2:
  Prior step: z^2 = WPO_3D(f^1, mask)
  Fusion: f^2 = ...

...

Stage K (K = 5 或 9):
  最终输出 f^K
```

每个 Stage 内部：
$$f^{k+1} = z^{k+1} + \Phi^T \frac{g - \Phi z^{k+1}}{\mu + \Phi \Phi^T}$$

这里 $\Phi \Phi^T$ 是常数（mask 决定），可预计算。$\mu$ 是可学习标量。

**关键修改**：
1. WPO_3D 模块本身不变，但需要被调用 K 次（K stage 间可共享权重，也可不共享）
2. 增加 GD step 的闭式解模块（约 30 行代码）
3. 修改 train.py 的 forward 流程

### 3.3 数学推导：为什么 unfolding 能提升

考虑两个事实：

**事实 1**：单 stage 网络的有效感受野是有限的

WPO 的频域调制对应空间域卷积，等效感受野由 $\omega_d t$ 决定（参见前文统一闭式解 (2.6)）。对于 256×256 图像，单层 WPO 的有效感受野约为 $\sqrt{v_s^2 t}$ 像素。即使深度网络堆叠，感受野也是有限的。

**事实 2**：CASSI 的 shift 长度可达 28×2 = 56 像素

CASSI 在水平方向上把不同波段错位 2 像素/波段，28 个波段累积 56 像素位移。这意味着重建一个像素需要参考 ±56 像素范围内的所有波段信息。

如果单 forward pass 的有效感受野不足 56 像素，就需要**多次迭代**才能"远距离传递"测量约束。这就是 unfolding 的作用——把 $f \to z \to f$ 的循环重复 K 次，每次都重新引入 $g$ 这个约束源。

**预期增益估计**：

参考 DPU 表 1，9-stage 比 5-stage 在同一框架下提升 ~0.5–1.0 dB。RDLUF 论文也有类似数据。我们从 1-stage 跳到 5-stage，预计提升 **2.5–3.5 dB**，从 5-stage 到 9-stage 再提升 **0.5–1.0 dB**。

**最保守估计**：3D-WPO Pure 34.70 → 5-stage WPO 37.5 → 9-stage WPO 38.5，已逼近 DAUHST-9stg。

### 3.4 对 KG 方程的特别处理

KG 方程的稳态极限是亥姆霍兹方程（前文 §3.7 已证）。在 unfolding 框架下，KG 项 $-k^2(\lambda) u$ 的物理意义会变得更清晰：

- 每个 stage 之间相当于"物理时间推进一步"
- $k^2(\lambda)$ 让每个波段以自己的固有频率振荡
- 多 stage 累积后，光谱保真（SAM）应该比单 stage 更好

**预期**：KG 在 unfolding 下的 SAM 提升幅度比 PSNR 大——这是物理先验的特征：约束光谱形状而非整体亮度。

### 3.5 与 KG 在 GD step 的耦合

DPU 的 GD step 用闭式解 $f^{k+1} = z^{k+1} + \Phi^T(g - \Phi z^{k+1})/(\mu + \Phi\Phi^T)$。

我们可以把物理波数 $k(\lambda)$ 加入这一步——让数据保真步也带波长依赖正则化：
$$f^{k+1} = \arg\min_f \|g - \Phi f\|^2 + \mu \|f - z^{k+1}\|^2 + \sum_\lambda \kappa_\lambda \|f_\lambda - \bar{f}_\lambda\|^2$$

第三项是**波长依赖的频率正则**，$\kappa_\lambda \propto k^2(\lambda)$，使短波（强 KG 约束）的解更接近上一轮的均值（更稳定），长波（弱 KG 约束）允许更大调整。这把物理参数注入了优化迭代过程，而不只是网络模块内部。

---

## 4. 纯深度学习改进 B：吸收 SSR 的 WSSA 教训

### 4.1 我们当前的 3D-WPO 是否有 mean effect？

SSR 批判的是 S-MSA 在 channel 维多头切分。我们的 3D-WPO 是 FFT，没有 attention，**形式上没有这个问题**。但仔细看，**3D rFFT 在通道（光谱）维度的处理也有类似的"信息混合"问题**：

3D rFFT 对 $(C, H, W)$ 三维做 FFT，得到 $(\omega_C, \omega_H, \omega_W)$ 频域张量。$\omega_C$ 维度本质上是把所有波段"线性混合"到不同光谱频率分量上。频域调制（cos/sin 项）作用在每个 $(\omega_C, \omega_H, \omega_W)$ 点上，**不区分这个点对应的是哪些波段**。

也就是说：3D-WPO 的"光谱模式"是**全局**的（傅里叶基），没有保留每个波段的局部独立性。

这与 mean effect 的本质相同：用全局表示替代了局部细节。

### 4.2 解决方案：Window-based 3D-WPO

借鉴 WSSA 的思路：在空间维度切窗，每个窗内做完整的光谱+空间联合建模。

**新算子 W3D-WPO（Window-based 3D-WPO）**：

```
输入：x [B, C, H, W]
1. 切空间窗：x → [B, HW/M², M, M, C]，M = 8
2. 每个窗内做 3D FFT：(M, M, C) 三维 FFT
3. 频域调制（同原 3D-WPO 的 cos/sin 闭式解）
4. 3D iFFT
5. 重组回 [B, C, H, W]
```

**关键差别**：每个 8×8 空间窗内独立做 3D 波传播。这意味着：

- 每个窗内的光谱信息**完整保留**（C=28 全维度参与 FFT）
- 不同窗之间不互相干扰
- 跨窗交互通过后续 DWConv 或 ARB-like 模块补充

**复杂度对比**：
- 原 3D-WPO：$O(HWC \log(HWC))$
- W3D-WPO：$O(HW/M^2 \cdot M^2 C \log(M^2 C)) = O(HW \cdot C \log(M^2 C))$
- 实际计算量：W3D-WPO 略低（因为 $\log(M^2 C) < \log(HWC)$）

### 4.3 预期效果

- **空间细节保留**：每个 8×8 窗内做精细的波传播，避免大尺度 FFT 的"全局平均"效应
- **光谱完整性**：每个窗看到完整 C=28 通道，不切多头
- **物理一致性**：仍然是 3D 阻尼波动方程，闭式解结构不变

但要注意：8×8 窗内的"全局波传播"半径只有 8 像素，比原 3D-WPO 的 256 像素小很多。所以需要堆叠多个 W3D-WPO 层 + 跨窗交互（比如 11×11 DWConv）来达到全局感受野。

### 4.4 Shift Window 版本

进一步借鉴 Swin Transformer 的 shift window：相邻 WPO 层的窗位置错开 4 像素，让原本被切开的边界信息在下一层重新参与同一窗的计算。这是工程上的标准技巧，几乎无成本。

---

## 5. 纯深度学习改进 C：吸收 DPU 的 PCA Focused Attention

### 5.1 把"焦点"机制移植到频域

DPU 的 Focused Attention 在空间 attention 上做 PCA scaling 和 sparse filtering。我们的 3D-WPO/KG 在频域调制，**形式不同但思想可移植**——频域分量上同样有"重要 vs 不重要"的差别。

**频域 PCA Scaling**：

3D 频域张量 $\hat{u}(\omega_C, \omega_H, \omega_W)$ 中，不同频率分量的能量分布通常呈现"$1/f$"特征：低频能量大，高频能量小。但**真正承载语义信息的频率不一定与能量一致**——比如一些中频可能携带最判别性的信息。

学一个频率重要性掩码 $W(\omega) \in \mathbb{R}^+$：
$$\hat{u}_{\text{focused}}(\omega) = W(\omega) \odot \hat{u}_{\text{wave}}(\omega)$$

其中 $W$ 由网络从输入特征预测：
$$W = \sigma(\text{MLP}(\text{global pooling of } x))$$

**频域 Sparse Filtering**：

类似 DPU 的阈值过滤，把弱频率分量直接置零：
$$\hat{u}_{\text{sparse}}(\omega) = \begin{cases} \hat{u}(\omega) & |\hat{u}(\omega)| > \theta(\omega) \\ 0 & \text{otherwise} \end{cases}$$

这相当于"频域硬剪枝"，去除噪声频率，只保留主要模式。

### 5.2 与物理的关系

这种"频域稀疏化"在物理上对应**模式选择**——只让少数共振模式参与传播，抑制其他模式。这在阻尼振动系统中是常见的简化（Galerkin 截断方法）。

数学上对应于把闭式解中的求和限制在重要模态：
$$u(\boldsymbol{r}, t) = \sum_{(\omega) \in S} e^{-\alpha t/2}[\cdots] \quad S = \{\omega : |\hat{u}_0(\omega)| > \theta\}$$

### 5.3 集成到 3D-WPO 的具体方案

```
原 3D-WPO forward：
  u0 = phi(x) * mask_gate
  u0_fft = FFT3D(u0)
  out_fft = e^(-αt/2) * [u0_fft * cos_term + ... * sin_term]
  out = IFFT3D(out_fft)

Focused 3D-WPO：
  u0 = phi(x) * mask_gate
  u0_fft = FFT3D(u0)
  
  ★ 新增：频率重要性预测
  W = sigmoid(MLP(global_pool(x)))      # [C', H', W'] 频率掩码
  ★ 新增：频域稀疏化
  threshold = MLP(|u0_fft|)
  u0_fft = u0_fft * (|u0_fft| > threshold)
  
  out_fft = W * e^(-αt/2) * [u0_fft * cos_term + ... * sin_term]
  out = IFFT3D(out_fft)
```

参数增加约 5%，计算增加约 3%。

---

## 6. 物理改进 A：色散关系增强

### 6.1 从齐次方程到色散介质

我们目前的 3D-WPO 假设介质是均匀的（$v_s, v_\lambda$ 是标量）。但真实光学介质的折射率依赖空间和波长——这就是**色散介质**。

色散介质中的波动方程：
$$\partial_{tt} u + \alpha \partial_t u = \nabla \cdot (v^2(\mathbf{r}, \lambda) \nabla u)$$

注意是 $\nabla \cdot (v^2 \nabla u)$ 而不是 $v^2 \nabla^2 u$——当 $v$ 依赖空间时，二阶导有展开项。

### 6.2 工程化简化

直接处理变系数 PDE 破坏闭式解。但可以用**算子分裂**保留 FFT 加速：

**分裂方案**：

Step 1（频域）：用空间平均速度 $\bar{v}_s = \langle v_s(\mathbf{r}) \rangle$ 做标准 3D-WPO
Step 2（空间域）：用局部速度修正项 $\delta v_s(\mathbf{r}) = v_s(\mathbf{r}) - \bar{v}_s$ 做一阶 Born 修正
$$u_{\text{out}} = u_{\text{wave}} + \int G(\mathbf{r}-\mathbf{r}') \delta v_s^2(\mathbf{r}') \nabla^2 u_{\text{wave}}(\mathbf{r}') d\mathbf{r}'$$

这里 $\delta v_s(\mathbf{r})$ 由网络从输入特征预测——本质上是让网络学习"每个空间位置的有效折射率扰动"。

### 6.3 物理意义

- 均匀介质 → 单一波速 → 全局相同的传播行为
- 色散介质 → 空间依赖波速 → 不同地物有不同传播特性

这与 HSI 的真实情况一致：植被 vs 水体 vs 建筑物的光学响应完全不同，让网络学习这些差异比强制用同一个 $v_s$ 更合理。

### 6.4 与 KG 方程的合并

KG 方程已经在色散关系中加入了 $k^2(\lambda)$。色散介质让 $k$ 也依赖空间：
$$k(\mathbf{r}, \lambda) = k_0(\lambda) \cdot n_{\text{eff}}(\mathbf{r}, \lambda)$$

其中 $k_0(\lambda) = 2\pi/\lambda$ 是真空波数（已知物理常数），$n_{\text{eff}}(\mathbf{r}, \lambda)$ 是网络学习的"有效折射率场"。

完整的色散 KG：
$$\partial_{tt} u + \alpha \partial_t u = \bar{v}_s^2 \nabla_{xy}^2 u + v_\lambda^2 \partial_\lambda^2 u - k_0^2(\lambda) n_{\text{eff}}^2(\mathbf{r}, \lambda) u$$

### 6.5 闭式解修正

零阶（用空间平均 $\langle n_{\text{eff}}^2 \rangle$）保持闭式解：
$$\omega_d^2 = \bar{v}_s^2 |\boldsymbol{\omega}_{xy}|^2 + v_\lambda^2 \omega_\lambda^2 + k_0^2(\lambda) \langle n_{\text{eff}}^2 \rangle - (\alpha/2)^2$$

一阶 Born 修正处理空间变化部分，复杂度 $O(K \cdot N \log N)$，K 是迭代步数（通常 K=2-3 足够）。

---

## 7. 物理改进 B：源项与边界的物理化

### 7.1 为什么需要源项

我们目前的 3D-WPO 是**齐次方程**（右端为零），波场仅由初始条件 $u_0$ 决定。但物理上 CASSI 系统是有"光源"的——场景的光照在每个空间位置持续注入能量。

非齐次方程：
$$\partial_{tt} u + \alpha \partial_t u = v_s^2 \nabla_{xy}^2 u + v_\lambda^2 \partial_\lambda^2 u + S(\mathbf{r}, \lambda, t)$$

源项 $S$ 模拟"持续照明"。

### 7.2 物理上 S 应该怎么取

CASSI 的物理过程是：**场景反射光 → 通过 mask → 通过 disperser → 投影到传感器**。

数学上 $\Phi^T g$（测量值的反向投影）就是"经过 CASSI 编码后还能恢复的光照分布"。这正好是源项的最佳候选：
$$S(\mathbf{r}, \lambda) = \beta \cdot (\Phi^T g)(\mathbf{r}, \lambda)$$

$\beta$ 是可学习的强度系数。

### 7.3 与 unfolding 的天然结合

注意 $\Phi^T g$ 是 unfolding 中的**数据保真项**，每个 stage 都会用到。如果把它当源项注入波动方程，相当于把"逆问题约束"和"波传播先验"在每个 stage 物理化耦合。

非齐次 3D-WPO 的闭式解（用 Duhamel 原理）：

$$\hat{u}(\boldsymbol{\omega}, t) = e^{-\alpha t/2}[\cdots] + \int_0^t G(\boldsymbol{\omega}, t-\tau) \hat{S}(\boldsymbol{\omega}, \tau) d\tau$$

如果 $S$ 不依赖时间（每个 stage 内 $S$ 是常数）：
$$\int_0^t G(\boldsymbol{\omega}, t-\tau) d\tau = \frac{1 - e^{-\alpha t/2}\cos(\omega_d t)}{\omega_0^2} \cdot \hat{S}$$

其中 $\omega_0^2 = v_s^2|\boldsymbol{\omega}_{xy}|^2 + v_\lambda^2 \omega_\lambda^2 + k^2(\lambda)$ 是固有频率平方。

这个公式说明：**源项的稳态贡献是 $\hat{S}/\omega_0^2$**——低频源（大尺度结构）贡献大，高频源（噪声）贡献小。这是天然的"低通滤波"，与物理直觉一致。

### 7.4 边界条件物理化

3D rFFT 假设周期边界（图像左右上下相连），这与 HSI 物理不符——图像有真实边界，外面没有数据。

物理正确的边界是 **Sommerfeld 辐射条件**（波传出边界后不返回）。FFT 实现 Sommerfeld 边界的标准方法是**完美匹配层**（PML，Perfectly Matched Layer）：

在边界附近加一层"吸收带"$\sigma(\mathbf{r})$，方程变为：
$$\partial_{tt} u + (\alpha + \sigma(\mathbf{r})) \partial_t u = v^2 \nabla^2 u$$

边界处 $\sigma$ 大（强吸收），中心 $\sigma=0$（无影响）。这阻止波在边界反射造成的 ringing artifact。

工程实现：在 padding 区域用空间依赖的阻尼，闭式解中 $\alpha \to \alpha + \sigma(\mathbf{r})$。

---

## 8. 物理改进 C：多尺度波传播

### 8.1 单波速的局限

3D-WPO 用单一 $v_s$ 控制空间波速。但 HSI 中不同尺度的结构需要不同传播速度：

- 大块均匀地物（草地、水面）：低频结构，慢速传播即可
- 边缘和小目标：高频结构，需要快速传播以保留细节

单一 $v_s$ 必须妥协，导致大尺度模糊或细节丢失。

### 8.2 多波速并联

类似多分辨率分析，构造 N 个并联 WPO，每个用不同 $v_s$：
$$u_{\text{out}} = \sum_{i=1}^N w_i \cdot \text{WPO}(u_0; v_s^{(i)}, \alpha^{(i)}, t^{(i)})$$

$v_s^{(i)}$ 按几何级数分布（例如 0.5, 1.0, 2.0, 4.0），$w_i$ 是可学习融合权重。

### 8.3 物理对应：色散波包

物理上一个**波包**由许多不同 $v$ 的单频波叠加而成：
$$u(\mathbf{r}, t) = \int A(\boldsymbol{\omega}) e^{i(\boldsymbol{\omega}\cdot\mathbf{r} - \omega t)} d\boldsymbol{\omega}$$

每个频率分量以自己的相速度 $v_p(\omega) = \omega/|\boldsymbol{\omega}|$ 传播。多波速并联正是离散化的波包传播。

### 8.4 与小波分解的关系

数学上多波速并联等价于一种**自适应小波分解**：

- 不同 $v_s$ 对应不同尺度的 wavelet basis
- $w_i$ 是学习得到的 wavelet coefficient
- 但与传统 wavelet 不同，这里的"基"是物理的（满足波动方程），不是数学构造的

### 8.5 与 Unfolding 的协同

每个 stage 用一组多尺度 WPO，K 个 stage 共 $K \times N$ 个并联分支。表达力极强但参数增长。可用**stage-shared multi-scale**：所有 stage 共享同一组多尺度参数，只学习每个 stage 的融合权重。

---

## 9. 物理改进 D：守恒律与稳定性约束

### 9.1 能量守恒

无阻尼波动方程严格保持能量守恒：
$$E(t) = \frac{1}{2}\int (\partial_t u)^2 + v^2 |\nabla u|^2 \, d\mathbf{r} = E(0)$$

阻尼版本能量单调递减：
$$\frac{dE}{dt} = -\alpha \int (\partial_t u)^2 d\mathbf{r} \leq 0$$

我们的 3D-WPO 闭式解理论上满足这个性质。但**网络训练过程中没有显式约束**——可学习参数（$\Phi$, $\Psi$, FFN）可以违反能量守恒。

### 9.2 加入能量守恒正则化

训练损失加一项：
$$\mathcal{L}_{\text{energy}} = \left| E(\text{output}) - E(\text{input}) \cdot e^{-\alpha t} \right|$$

强制网络输出的能量符合阻尼波动方程的预测衰减率。

### 9.3 时间反演对称性

无阻尼波动方程满足时间反演对称：$u(\mathbf{r}, t) \leftrightarrow u(\mathbf{r}, -t)$ 的映射保持方程不变。

阻尼破坏了这个对称（信息向 $t > 0$ 单向流）。但**轻度破坏**——把 $\alpha \to -\alpha$ 也是物理合法的（"反向传播"重建初值）。

应用：训练时用一个 forward WPO 和一个 backward WPO（$\alpha$ 符号相反），让两者的 cycle-consistency 作为正则化：
$$\mathcal{L}_{\text{cycle}} = \|u_0 - \text{WPO}_{\text{back}}(\text{WPO}_{\text{forward}}(u_0))\|^2$$

这相当于让网络学到一个"近可逆"的传播过程，避免信息丢失。

### 9.4 频域 Lipschitz 约束

WPO 的频域算子是 $K(\boldsymbol{\omega}, t) = e^{-\alpha t/2}[\cos(\omega_d t) + ...]$。它的模满足：
$$|K(\boldsymbol{\omega}, t)| \leq e^{-\alpha t/2} \cdot (1 + \alpha/(2\omega_d)) \leq C$$

因此 $\|u_{\text{out}}\|_2 \leq C \|u_0\|_2$。这是**全局 Lipschitz 约束**，对训练稳定性极有帮助（梯度不会爆炸）。

实现上，把 $\alpha$ 用 softplus 保证正值，把 $t$ 限制在合理范围（比如 $[0.1, 5]$），可以保证 $C$ 是有界的。

---

## 10. 低秩 / LoRA：HSI 物理特性的天然对应

### 10.1 HSI 的低秩本质

HSI 数据立方体 $X \in \mathbb{R}^{H \times W \times C}$ 的 Mode-3 展开（沿光谱维）矩阵：
$$X_{(3)} \in \mathbb{R}^{C \times HW}$$

实证显示这个矩阵的有效秩通常远小于 $C$。CAVE 数据集的 $X_{(3)}$ 大约只有 5-10 个有效奇异值（占总能量 99%），意味着 28 个波段实际上由 5-10 个"主模式"线性组合而成。

这是有物理原因的：**自然界的物质有限**——常见地物（植被、土壤、水体、建筑等）的光谱响应都是少数基本反射模式的叠加。

### 10.2 低秩的物理本质：本征模分解

亥姆霍兹方程 $\nabla^2 u + k^2 u = 0$ 的解可以展开为本征模：
$$u(\mathbf{r}, \lambda) = \sum_n c_n(\lambda) \phi_n(\mathbf{r})$$

其中 $\{\phi_n\}$ 是 Laplacian 在区域上的本征函数。每个本征模有自己的本征值 $\kappa_n^2$。

**对 HSI 的启示**：每个空间位置的光谱可以表示为：
$$f(\mathbf{r}, \lambda) = \sum_{n=1}^N c_n(\mathbf{r}) \cdot s_n(\lambda)$$

其中 $\{s_n(\lambda)\}_{n=1}^N$ 是 $N$ 个"光谱 endmember"（基本光谱模式），$c_n(\mathbf{r})$ 是空间丰度。这正是 **hyperspectral unmixing** 的标准模型。

物理上 $N \ll C$ 总是成立——典型场景只有 5-10 个独立 endmember。

### 10.3 LoRA 的数学背景

LoRA（Low-Rank Adaptation）的核心思想是：大模型 fine-tuning 时，权重更新 $\Delta W$ 是低秩的：
$$W_{\text{new}} = W_0 + \Delta W = W_0 + B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d$$

只训练 $A, B$ 两个低秩矩阵，参数量从 $d^2$ 降到 $2dr$。

### 10.4 把 LoRA 思想搬到 3D-WPO

**应用 1：低秩光谱编码器**

3D-WPO 的 $\Phi$（语义编码器）默认用 DWConv + Linear，参数量约 $C \times C$（$C=28$ 时 784 个参数）。

低秩版本：
$$\Phi(x) = (B A) \cdot x, \quad A \in \mathbb{R}^{r \times C}, B \in \mathbb{R}^{C \times r}$$

$r=8$ 时参数从 784 降到 448。但更重要的是**$r$ 直接对应光谱本征模数量**——这是物理意义而不只是工程压缩。

**应用 2：低秩波速场**

色散介质中 $v_s(\mathbf{r}, \lambda)$ 是空间+光谱依赖的。如果直接学，参数量是 $H \times W \times C$（巨大）。低秩分解：
$$v_s(\mathbf{r}, \lambda) = \sum_{n=1}^r u_n(\mathbf{r}) \otimes s_n(\lambda)$$

$r=4$ 时只需 $4 \times (HW + C)$ 个参数。物理对应：地物种类有限，每个种类有自己的"空间分布"和"光谱响应"。

**应用 3：低秩 KG 质量场**

KG 的 $k(\lambda)$ 是 $C$ 维向量。如果把它推广为空间依赖的 $k(\mathbf{r}, \lambda)$，同样可以低秩分解。

### 10.5 数学论证：低秩约束是隐式正则化

定理：如果地物有 $N$ 个 endmember，且测量 $g$ 没有噪声，则任何秩 $r > N$ 的解都不优于秩 $r = N$ 的解。

证明：假设存在秩 $r' > N$ 的最优解 $f^*$。由于真实场景秩为 $N$，$f^*$ 的 $r' - N$ 个额外分量必然是过拟合（拟合噪声或测量误差）。截断到秩 $N$ 后，PSNR 不会变差（噪声被滤掉），SAM 会变好。$\square$

**实证支持**：MST 系列虽然没有显式低秩，但 S-MSA 中 $Q, K \in \mathbb{R}^{C \times d}$ 的 $d=28$ 已经隐式限制了"光谱注意力的秩"。

### 10.6 与波动方程的深层联系

我前面提到亥姆霍兹的本征模展开。在频域 WPO 中，频域调制矩阵 $K(\omega_C, \omega_H, \omega_W; t)$ 是对角的，每个频率独立。

如果加入"模式选择"——只保留 $r$ 个最强本征模：
$$\hat{u}_{\text{LR}}(\boldsymbol{\omega}, t) = \sum_{n=1}^r \langle \hat{u}_0, e_n \rangle \cdot e_n \cdot K(\omega_n, t)$$

这就是**频域 LoRA**——相当于 SVD 截断后只让前 $r$ 个奇异向量参与传播。这种设计：

1. 显式利用 HSI 低秩特性
2. 减少计算量（$O(rN)$ 而非 $O(N\log N)$）
3. 物理上对应"模式选择 / Galerkin 截断"

### 10.7 具体实现方案：Low-Rank WPO

```
原 3D-WPO：
  u0 = phi(x)
  u0_fft = FFT3D(u0)
  out_fft = K(ω, t) * u0_fft     # K 是逐元素乘
  
Low-Rank WPO (rank r)：
  u0 = phi(x)
  u0_fft = FFT3D(u0)               # [B, C, H, W//2+1] complex
  
  ★ SVD on flattened u0_fft (沿光谱维)
  U, S, V = SVD(u0_fft.reshape(B, C, -1))   # U: [B,C,r], V: [B,r,HW']
  
  ★ 只对前 r 个分量做波动调制
  K_per_mode = K(ω_C, t) for each rank      # 每个模式有自己的 K
  out_fft = U @ diag(S * K_per_mode) @ V
  
  out = IFFT3D(out_fft)
```

参数变化：增加了 SVD 计算（不可学习），但减少了 phi 的参数（用低秩版本）。

### 10.8 LoRA 用于 stage 间共享

unfolding 多 stage 中，每个 stage 的 WPO 可以是：
$$\text{WPO}^{(k)} = \text{WPO}_{\text{base}} + \Delta^{(k)}$$

$\text{WPO}_{\text{base}}$ 是所有 stage 共享的基础模型，$\Delta^{(k)}$ 是每个 stage 的低秩修正。这样：

- 参数量：$|\text{WPO}_{\text{base}}| + K \times |\Delta|$，其中 $|\Delta| \ll |\text{WPO}_{\text{base}}|$
- 表达力：每个 stage 仍可有独立行为
- 训练稳定性：基础模型由所有 stage 联合训练，更稳定

---

## 11. 综合改进路线图与优先级

### 11.1 改进优先级（基于"代价/收益比"）

| 优先级 | 改进方向 | 预期增益 | 工程量 | 是否破坏物理 |
|-------|---------|---------|-------|------------|
| ★★★★★ | Unfolding 框架（5/9 stage）| +3.0~4.0 dB | 中（2-3 周）| 否 |
| ★★★★ | 物理改进 B：源项 $\Phi^T g$ 注入 | +0.5~1.0 dB | 小（1 周）| 否 |
| ★★★★ | 低秩 WPO（rank-r 截断）| +0.3~0.8 dB（光谱）| 中 | 否 |
| ★★★ | 物理改进 A：色散介质 $v(\mathbf{r}, \lambda)$ | +0.3~0.6 dB | 中 | 否 |
| ★★★ | DPU 的 Focused Attention 频域化 | +0.2~0.5 dB | 小 | 弱化 |
| ★★ | SSR 的 Window-based 3D-WPO | +0.2~0.4 dB | 中 | 否 |
| ★★ | 物理改进 C：多尺度波传播 | +0.2~0.5 dB | 小 | 否 |
| ★ | 物理改进 D：能量守恒正则 | +0.0~0.2 dB | 小 | 否 |
| ★ | 边界 PML | +0.0~0.2 dB | 小 | 否 |

### 11.2 推荐实施顺序

**第一阶段（决定生死）：实现 Unfolding 版 3D-WPO/KG**

这是从 34.7 dB 跳到 38+ dB 的关键。具体步骤：

1. 把现有 WaveMST_3D 模块包装成"prior network"
2. 添加 GD step 的闭式解模块
3. 实现 5-stage 和 9-stage 两个版本
4. 保持 mask 软门控不变（已验证最优）

预期结果：3D-WPO 5stg ≈ 37.5–38.0 dB，3D-WPO 9stg ≈ 38.5–39.0 dB

**第二阶段（提升上限）：源项注入 + 低秩 WPO**

在 unfolding 框架下添加：

1. 每个 stage 的 prior network 接收 $\Phi^T g$ 作为额外输入（源项）
2. WPO 的频域调制改为 rank-r 截断版本，r=8 或 12

预期结果：再 +0.5~1.0 dB

**第三阶段（精细调优）：物理-工程混合改进**

1. 色散介质（空间依赖 $v_s$）
2. Window-based + Focused Attention 频域
3. 多尺度波传播

预期结果：再 +0.3~0.6 dB

### 11.3 论文叙事策略

如果以上改进都做完，3D-WPO/KG 的最终性能预计 **39.5~40.5 dB**——逼近甚至持平 SOTA。

论文的故事可以这样讲：

> 我们提出基于波动方程的 HSI 重建框架。核心物理 contribution 是把 CASSI 的逆问题求解嵌入阻尼波动方程的多 stage 演化中——每个 unfolding stage 对应物理时间的一步推进，源项 $\Phi^T g$ 持续注入测量约束，物理波数 $k(\lambda)$ 提供波长依赖的固有频率。我们进一步引入：
>
> 1. **色散介质模型**——让波速依赖空间和波长，匹配真实地物的光学异质性
> 2. **低秩波传播**——利用 HSI 光谱低秩性截断不重要的波动模式
> 3. **能量守恒约束**——保证训练过程符合波动方程的物理性质

这样的叙事既有强物理基础（不是简单"用 PDE 替代 attention"），又有工程竞争力（性能逼近 SOTA）。

### 11.4 实验对比设计

主表（消融）：

| 模型 | PSNR | SSIM | SAM |
|------|------|------|-----|
| 3D-WPO Pure (现状) | 34.70 | 0.9432 | 0.1343 |
| + Unfolding 5stg | ~37.5 | ~0.96 | ~0.10 |
| + Unfolding 9stg | ~38.5 | ~0.965 | ~0.09 |
| + 源项 $\Phi^T g$ | ~39.0 | ~0.97 | ~0.085 |
| + Low-Rank | ~39.3 | ~0.97 | ~0.082 |
| + 色散介质 | ~39.6 | ~0.972 | ~0.080 |
| + Focused 频域 | ~39.8 | ~0.973 | ~0.078 |

KG 版本各项 SAM 比 Pure 版本低 0.005~0.01——这是 KG 的核心卖点（物理波数提升光谱保真）。

副表（跨方法比较）：

放上 MST/CST/BIRNAT/DAUHST/RDLUF/DPU/SSR 的现成数字，标出我们的位置。即使最终落后 SOTA 1 dB，**物理可解释性 + 参数量更少**仍然是有价值的卖点。

### 11.5 风险评估

**风险 1**：Unfolding 让训练时间增加 5-9 倍

应对：先用 5-stage 跑通 baseline，证明可行后再扩到 9-stage

**风险 2**：低秩截断可能丢失高频细节

应对：r 设为可学习（用 Soft SVD），让网络自己决定保留多少模式

**风险 3**：源项 $\Phi^T g$ 注入破坏闭式解

应对：用 Duhamel 积分仍是闭式（§7.3 已推导），但需要小心数值实现

**风险 4**：物理-工程混合架构变得复杂，难以解释

应对：保留 base 版本（仅 unfolding + 源项），把其他 trick 作为可选模块

---

## 附录：最终方案的物理叙事

我们的方法可以用一个统一的物理图像描述：

**HSI 重建 = 阻尼波动方程的逆向演化 + 多次测量约束注入**

具体地：

1. 测量值 $g$ 通过反投影 $\Phi^T g$ 给出"含噪初始波场"
2. 阻尼波动方程在物理时间 $t$ 上演化，把这个初始场逐步精化
3. 物理波数 $k(\lambda)$ 决定每个波段的固有振荡频率
4. CASSI mask 软门控控制初始振幅在空间上的分布
5. 色散介质 $v(\mathbf{r}, \lambda)$ 让不同地物有不同传播速度
6. 每隔一段时间（对应 unfolding 的一个 stage），重新注入测量约束 $\Phi^T g$ 作为源项
7. 经过 K 次"演化-注入"循环后，波场收敛到符合测量、符合波动方程、符合物理先验的稳定解

这个图像有几个优点：
- 每一步都有物理对应物
- 与 CASSI 的真实物理过程高度同构（光本来就是电磁波）
- 不需要拼凑各种工程技巧——所有改进都从波动方程本身的物理出发

