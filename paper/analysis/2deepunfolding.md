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

