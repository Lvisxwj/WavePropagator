# problem.md — SMILE² 论文 Problem 部分草稿

> **覆盖范围**：Problem Description · Observation · Gap Analysis · Key Insight · Assumptions
> **写作风格**：参照 CVPR/ICCV/NeurIPS 顶会 Section 1（Introduction + Problem Setup）+ Section 3.1（Preliminaries）的组织方式；保留多于实际行文所需的素材，便于后续删改。
> **基准命名**：以 `name_mapping.md` 为准（SMILE² / SWAP / MI / AdaSpec / KGD / LDE / SEC / DAG / NLE / LRB / A-HQS）。

---

## 0. 速记摘要（供 Introduction 起笔用）

> Compressive spectral imaging seeks to reconstruct a 3D hyperspectral cube
> $f \in \mathbb{R}^{H \times W \times \Lambda}$ from a single 2D coded snapshot. Existing
> deep-unfolding solvers either rely on heavy spatial-spectral Transformers
> whose attention disregards the *physical structure* of the coding pipeline,
> or invoke physics-flavoured operators (heat / wave equations) that propagate a
> pre-CASSI-degraded initial field, in effect amplifying the degradation.
> We argue that **the bottleneck is not the propagator but its input**: the
> coded-aperture mask, dispersive shift and integrative compression must be
> explicitly disentangled *before* any global-propagation prior is applied,
> and the propagator must remain physically grounded so that the resulting
> system inherits convergence guarantees from accelerated optimisation. SMILE²
> realises this idea with a tightly-coupled estimation-evolution pair (LDE
> ↔ SWAP) wrapped in an A-HQS unfolding with Nesterov acceleration.

---

## 1. Problem Description

### 1.1 Forward (Physical) Model

CASSI 的物理过程可写成

$$g(x, y) = \sum_{\lambda=1}^{\Lambda} M(x, y) \cdot f\!\big(x - d(\lambda),\, y,\, \lambda\big) + n(x, y), \tag{1.1}$$

其中

- $f(x,y,\lambda) \in \mathbb{R}^{H\times W\times \Lambda}$ —— 待重建的高光谱立方体（CAVE 设定下 $\Lambda = 28$，$H{=}W{=}256$）；
- $M(x, y) \in [0, 1]^{H\times W}$ —— **物理掩模**（编码孔径），沿空间有结构、沿光谱方向**不调制**；
- $d(\lambda)$ —— 棱镜/光栅引入的色散偏移，$d(\lambda)=\mathrm{step}\cdot\lambda$（CAVE 中 $\mathrm{step}=2$ px）；
- $n(x, y)$ —— 传感器噪声；
- $g \in \mathbb{R}^{H \times W'}$，$W' = W + (\Lambda - 1)\cdot\mathrm{step}$ —— 单帧 2D 测量。

矩阵化记号：

$$g = \Phi f + n, \qquad \Phi \in \mathbb{R}^{HW' \times HW\Lambda},$$

其中 $\Phi$ 复合了 *masking* → *spectral shifting* → *spectral summation* 三种线性退化。CASSI 的逆问题严重欠定：测量维 $HW'$ 远小于待求维 $HW\Lambda$，压缩比约 $1/\Lambda$。

### 1.2 Inverse-Problem Formulation

经典做法：

$$\min_{f} \;\; \tfrac{1}{2}\big\|g - \Phi f\big\|_2^2 \;+\; \gamma\,\mathcal{R}(f), \tag{1.2}$$

其中 $\mathcal{R}$ 是图像先验（TV, sparsity, 低秩, 学习先验等）。Deep Unfolding 将 (1.2) 的迭代算子展开成 $K$ 个 stage，每个 stage 串联 *data-fidelity step*（GD/HQS/ADMM）与 *learnable prior*。

### 1.3 学术目标

> 在保持 CASSI 物理一致性的前提下，构造一个能 **同时建模成像退化** 与 **全局光谱-空间传播** 的轻量、稳定、可收敛的深度展开框架，使得：
>
> 1. **退化感知**：每个 stage 显式估计 sensing error $\Delta \Phi$、空间退化权重 $w$、噪声水平 $\sigma$；
> 2. **物理传播**：以 3D 阻尼波动方程的闭式解作为先验算子，复杂度 $O(N\log N)$；
> 3. **加速收敛**：基础迭代格式提供 $O(1/K^2)$ 的全局收敛速率；
> 4. **轻量化**：参数量与 FLOPs 显著低于 SOTA Transformer-based unfolding（DPU/SSR/RDLUF）。

---

## 2. Observations（实验 + 文献证据）

> 这一节给读者一组 **可验证的事实**，作为后续 Insight 的支撑。每条观察都配「证据」与「论文里如何呈现」。

### 2.1 Observation A —— 单帧 end-to-end Transformer 性能瓶颈

| 方法 | PSNR / dB | Params / M |
|------|-----------|------------|
| MST (CVPR 2022, e2e) | 35.18 | 2.03 |
| CST (ECCV 2022, e2e) | 36.12 | 3.00 |
| BIRNAT (e2e, RNN) | 37.58 | 4.40 |

**事实**：在不引入数据保真重注入的 e2e 设定下，PSNR 普遍 $<$ 38 dB；参数量进一步加大边际收益递减。
**论文里如何用**：「Observe that e2e variants saturate around 37 dB; the gap to deep unfolding is **structural**, not capacity-bound.」

### 2.2 Observation B —— Unfolding 的边际收益远高于堆 ML 层

来自本工程对 3D-WPO（SWAP 的早期版本）的消融（@60 epoch）：

| 改动 | 参数 Δ | PSNR Δ | dB / M |
|------|--------|--------|---------|
| 5-stage GAP unfolding | +0.06 M | **+3.10** | **+31.0** |
| WSSA 注意力增强 | +3.01 M | +1.16 | +0.39 |
| FBA（小波分解） | +3.68 M | +0.84 | +0.23 |

**事实**：每 M 参数，「展开」的 PSNR 增益约是「ML 堆砌」的 **80×**。
**论文里如何用**：「Unfolding contributes structural gains (data-fidelity re-injection) that no amount of feature stacking can replicate.」

### 2.3 Observation C —— SOTA Unfolding 的关键收益来自“退化建模”

DPU (CVPR 2024) 自身的消融（Table 3）：

| 配置 | PSNR | 增益来源 |
|------|------|---------|
| Baseline (无 DPF) | 37.28 | — |
| + Focused Attention | 38.49 | +1.21（attention） |
| + Intuitive DPF | 38.76 | +0.27 |
| + Basic DPF | 39.23 | +0.47 |
| + Full DPF (双 prior 融合) | 39.62 | +0.39 |
| **DPF 累计** | | **+1.13（退化建模）** |

**事实**：退化建模与注意力机制对 DPU 的贡献几乎对等（+1.13 vs +1.21）。
**论文里如何用**：「Half of DPU's improvement comes from explicit degradation modelling, yet most physics-guided unfolding works ignore this branch.」

### 2.4 Observation D —— $\Phi^{\!\top} g$ 是“脏”的初始场

$$\Phi^{\!\top} g = \Phi^{\!\top}\Phi\,f_{\text{GT}} + \Phi^{\!\top} n.$$

$\Phi^{\!\top}\Phi$ 并非恒等：mask 透射率非均匀（典型 50%），色散偏移导致空间错位，求和导致 $\Lambda$ 个波段在 2D 上叠加；因此初始场带有**调制伪影 + 错位 + 混叠 + 放大噪声**四重退化。**事实**：直接把这个初始场喂给任何全局算子（attention / FFT / Mamba），都会把局部错误**扩散到全局**。

### 2.5 Observation E —— 阻尼波动方程的频域闭式解 *存在* 且 *稳定*

对 3D 阻尼波方程做 Fourier 变换，每个 $(\omega_x, \omega_y, \omega_\lambda)$ 是独立二阶常系数 ODE：

$$\hat u_t = e^{-\alpha t/2}\!\left[\hat u_0\,\mathrm{Cs}(\eta, t) + \big(\hat v_0 + \tfrac{\alpha}{2}\hat u_0\big)\mathrm{Sn}(\eta, t)\right], \quad \eta = \omega_0^2 - (\alpha/2)^2.$$

能量泛函满足 $dE/dt = -\alpha\,|\partial_t\hat u|^2 \le 0$ —— 系统**全局稳定**；振荡区保留高频（边缘、纹理），衰减区抑制低频背景，与 HSI 任务诉求天然对齐。

### 2.6 Observation F —— Nesterov 动量在不动点迭代上是“免费午餐”

Phy-CoSF (2026) 和 CA²UN (IET 2025) 已经实证：在 HQS unfolding 上加入

$$\hat z^k = z^k + \beta_k(z^k - z^{k-1}), \qquad \beta_k = \sigma(\theta_k) \in (0, 1),$$

只需新增 $K$ 个标量参数即可将收敛速率从 $O(1/K)$ 提升到 $O(1/K^2)$，PSNR 稳定 +0.3 ~ +0.5 dB。**事实**：动量是**几乎零参数**的结构性改进。

### 2.7 Observation G —— CASSI mask 在光谱方向是常量

由 (1.1)，$M(x,y)$ 沿 $\lambda$ 不变。等价地，$\hat M(\omega_x,\omega_y,\omega_\lambda) = \hat M_{xy}(\omega_x,\omega_y) \cdot \delta(\omega_\lambda)$。**事实**：mask 注入只在空间频率上扰动，不破坏 3D 波方程在光谱频率上的可分性，因此 “乘后 FFT” 与 “FFT 后空间卷积” 数学等价，**$O(N\log N)$ 保持**。

---

## 3. Gap Analysis（已有方法的不足）

按“四类方法 × 五个维度”交叉对比：

| | Data fidelity | Mask / Degradation modelling | Global prop. | 复杂度 | 收敛保证 |
|---|---|---|---|---|---|
| e2e Transformer (MST/CST) | ✗（仅初始化） | 弱（mask concat） | ✓ S-MSA | $O(C^2 HW)$ | ✗ |
| GAP-Net 系列 | ✓（GD） | ✗ | ✗ | $O(HW \Lambda)$ | $O(1/K)$ |
| RDLUF / DAUHST | ✓ | 隐式（DAN） | ✓ attention | $O(C^2 HW)$ | $O(1/K)$ |
| DPU (DPF + Focused) | ✓ | 显式（DPB） | ✓ Focused-Attn | $O(C^2 HW)$ | $O(1/K)$ |
| SSR (WSSA + ARB) | ✓ | 局部 | ✓ window-Attn | $O(C^2 HW)$ | $O(1/K)$ |
| WaveFormer 类 (e2e) | ✗ | ✗ | ✓ 3D FFT | $O(N \log N)$ | — |
| **SMILE² (本工作)** | ✓ + Φ_eff | ✓ SEC+DAG+NLE | ✓ SWAP（物理） | $O(N \log N)$ | **$O(1/K^2)$** |

### 3.1 Gap 1 — “物理传播 ⊕ 退化建模” 从未真正同框

WaveFormer / Heat-former 给出物理算子但不带 unfolding；DPU/SSR 给出强 unfolding 但 prior 仍是工程化 Transformer。**没有方法同时**做到（i）显式退化估计、（ii）频域闭式物理传播、（iii）二阶加速展开。SMILE² 正好填这一格。

### 3.2 Gap 2 — Mask 的“先验角色”被严重简化

主流做法把 mask 当作 attention 的额外通道或乘法常量，忽略了 *spatial-only* 这一物理事实（Obs. G）。SMILE² 通过 **MI**（Modulated Initialization）把 mask 注入到波方程的初始振幅，把 (1.1) 中 $M$ 的物理位置完整保留。

### 3.3 Gap 3 — 噪声水平没有反馈到“传播强度”

DPU/SSR 的 prior network 都是噪声盲；但波方程的阻尼系数 $\alpha$ 直接控制传播保守度——噪声大时应高阻尼（保守）、噪声小时应低阻尼（保留高频）。SMILE² 通过 **NLE**（$\sigma > 0$）将噪声水平耦合到 $\alpha_{\text{eff}} = \alpha + \lambda_\sigma \sigma$（**Estimation-Evolution** 名字由此而来）。

### 3.4 Gap 4 — Wiener 滤波从未与展开结合

频域中信号-噪声功率比（SNR）天然给出最优滤波器（Wiener）。SMILE² 的 **AdaSpec** 把 Wiener 表达式

$$W(\boldsymbol\omega) = \sigma_{\text{gate}}\!\left(\frac{|\hat u_0|^2 - \sigma^2}{|\hat u_0|^2 + \sigma^2 + \epsilon}\right) \in [0,1]$$

嵌入 WPO 的频域调制中，**零额外参数**复用 NLE 输出的 $\sigma$。

### 3.5 Gap 5 — 收敛速率与稳定性鲜有联合保证

之前的物理 unfolding（Phy-CoSF, CA²UN）虽用 A-HQS，但 prior 不是物理算子，能量稳定性需经验保证。SMILE² 的 SWAP 自带 $\frac{dE}{dt} \le 0$（Obs. E），与 A-HQS 的 $O(1/K^2)$ 共同构成**双层稳定性 + 加速**。

### 3.6 Gap 6 — 参数效率与 SOTA 不匹配

| 方法 | PSNR (5stg) | Params |
|------|-------------|--------|
| DPU-5stg | 39.62 | 1.59 M |
| RDLUF (≈5stg) | 39.57 | 1.81 M |
| SSR-S | 39.19 | 1.73 M |
| **SMILE²-5stg (shared)** | 目标 39+ | **~0.90 M** |

SMILE² 的物理算子参数严格独立于 $H, W$（仅 5 个标量），把额外参数花在 LDE/LRB 而非 attention 投影上。

---

## 4. Key Insights（论文卖点的种子）

> 这里列出 5 个 insight；任何一个单独都可以作为 "Our key insight is …" 的题眼。

### 4.1 Insight 1 — “The propagator is fine; its input is the problem.”

The performance ceiling of 3D-WPO is set not by *how well it propagates*, but by **how degraded its initial condition is**.
在 (1.2) 框架中，prior $\mathcal{R}(f)$ 通常被建模为 denoiser；但 wave equation 的物理角色是 **propagator**（保结构、做长程交互），让它去“去噪”天生错位。**正解**是把“去退化”交给前置模块（LDE），把“传播”留给波方程（SWAP）。

### 4.2 Insight 2 — Estimation-Evolution 的双向耦合（E² 命名由来）

LDE → SWAP **不是单向的“清理 → 传播”**；NLE 输出 $\sigma$ 反过来调制 SWAP 的阻尼 $\alpha_{\text{eff}}$，构成

$$\sigma \xrightarrow{\text{NLE}} \alpha_{\text{eff}} \xrightarrow{\text{SWAP}} f \xrightarrow{\text{NLE}} \sigma' \ldots$$

的闭环。第二次估计的 $\sigma$ 由更干净的 $f$ 给出，更准确；这一“估计⇄演化”反复发生在每个 stage 内部，是 **SMILE² 中“E²”的含义**。

### 4.3 Insight 3 — Mask is a spatial-only Dirac in $\omega_\lambda$

由 Obs. G，$M(x,y)$ 的 3D Fourier 是 $\hat M_{xy}(\omega_x,\omega_y)\,\delta(\omega_\lambda)$。这一观察让 MI 的实现 **只需在空间域乘 $M$ 即可**，FFT 之后无需做卷积，闭式解 (Eq. 2.6) 形式不变。光谱可分性是 SWAP 复杂度优势 ($O(N\log N)$) 的根基。

### 4.4 Insight 4 — Wiener 滤波 = AdaSpec：物理 + 统计的“免费缝合”

频域 Wiener 表达式在统计上最优；只要给定 $\sigma$（NLE 已经提供）和 $|\hat u_0|^2$（每个 stage 都要算），AdaSpec 的形式

$$W(\boldsymbol\omega) = \sigma_{\text{gate}}\!\left(\frac{|\hat u_0|^2 - \sigma^2}{|\hat u_0|^2 + \sigma^2 + \epsilon}\right)$$

**不引入任何可学参数**就能在 WPO 频域内完成 SNR 自适应频带加权。这是“物理先验 × 统计先验”的零成本融合。

### 4.5 Insight 5 — Nesterov 在不动点迭代上是结构性礼物

A-HQS 中 $\hat z^k = z^k + \beta_k (z^k - z^{k-1})$ 只新增 $K$ 个标量（$K \approx 5$），但把收敛阶从 $O(1/K)$ 提到 $O(1/K^2)$，等价于 “**5 stage 干 9 stage 的活**”。这是 SMILE² 把参数预算花在 LDE/SEC/DAG/NLE/LRB 而非更多 stage 上的依据。

---

## 5. Assumptions（明确写出，便于审稿人验真）

> 把假设列清楚，比含糊带过更经得起 rebuttal。

### A1 — 物理掩模沿光谱不变
$M(x, y, \lambda) = M(x, y)$。  现实成立（编码孔径是一块物理板）。

### A2 — 色散偏移线性可知
$d(\lambda) = \mathrm{step}\cdot\lambda$，且 $\mathrm{step}$ 已知。CAVE 数据集设定 $\mathrm{step}=2$ px。

### A3 — 噪声近似加性高斯
$n \sim \mathcal{N}(0, \sigma^2 I)$。$\sigma$ 由 NLE 估计；当噪声偏离高斯（如泊松-高斯混合），NLE 给出的是“等效高斯标准差”。

### A4 — 待重建立方体 $f \in [0, 1]^{H\times W\times \Lambda}$
为了 PSNR/SSIM 的 0-1 归一化口径与 CASSI/MST 系列对齐。

### A5 — 波方程参数全局共享但 stage 间独立
每个 SWAP 实例的 $(\alpha, v_s, v_\lambda, t, \lambda_\sigma)$ 都是标量（5 个 float），可学；展开 stage 之间可共享或独立（`SHARE_STAGE_WEIGHTS` 开关）。

### A6 — Borns 近似仅用于 KGD 可选分支
若启用 **KGD**（Klein-Gordon Dispersion，可选），假设质量场 $m_0^2 \le 0.5$，使 Born 一阶展开误差 $O((m_0^2 t / \omega_0^2)^2)$ 可控（详见 algorithm.md §KGD）。

### A7 — Nesterov 动量系数 $\beta_k \in (0, 1)$
$\beta_k = \sigma(\theta_k)$，$\theta_k$ 可学；初始化 $\theta_k = 0$ 对应 $\beta_k = 0.5$，对应 Nesterov 经典启动方式。

### A8 — FFT 维度 pad 到 2 的幂
为了 cuFFT 性能稳定。`FFT_PAD_TO_POW2 = True`；不改变数学结果（在 frequency support 外补零）。

### A9 — $\Phi\Phi^{\!\top}$ 是常数
在 GD step 闭式解中 $\Phi\Phi^{\!\top}$ 仅依赖 mask，可预计算并缓存（`compute_PhiPhiT`）；该假设对所有 CASSI unfolding 通用。

### A10 — 训练目标为 RMSE + 多 stage 加权
$\mathcal{L} = \sum_{k=1}^{K} w_k \cdot \text{RMSE}(f^k, f_{\text{GT}})$，权重 $w_K = 1$ 且 $w_{K-1}{=}0.7,\, w_{K-2}{=}0.5,\, w_{K-3}{=}0.3$；非主 stage 受较弱监督，避免“过早收敛到中间解”。

---

## 6. 与参考论文的取舍

| 参考论文 | 借用机制 | SMILE² 的修改 |
|---------|---------|---------------|
| WaveFormer (Wave equation 视觉建模) | 3D 阻尼波方程频域闭式解 | 改成各向异性、补全过阻尼分支、加入 $\sigma$ → $\alpha$ 耦合 |
| Heat-former | 频域算子先验 | 对照组（论文里作为 ablation：换阻尼波 → 热方程，证明 oscillatory 区比 monotone 衰减更重要） |
| MST / CST | mask-aware unfolding | 保留 ParaEstimator、U-Net 骨架；把 S-MSA 全部替换为 WPO3D Block |
| DPU (CVPR 2024) | 退化建模 + 双 prior | LDE 借用 DPB 的“mask 差异”思路，再叠加 DERNN-LNLT 的 $\Delta\Phi$/$\sigma$；放弃 PCA Focused Attention（与 FFT 重叠） |
| SSR (CVPR 2024) | Mean-effect 分析 + WSSA | 启发 W-SWAP（可选 Swin 窗口 WPO），但不当默认 |
| Phy-CoSF / CA²UN | A-HQS + Nesterov 动量 | 直接采用动量项；GD step 嵌入 $\Phi_{\text{eff}} = \Phi + \Delta\Phi$（SEC）作为新点 |
| RDLUF | 显式退化表示 | LDE 的 SEC 部分对应；其余（spatial-spectral mixing）被 SWAP 频域闭式解替代 |

---

## 7. Risks & Limitations（写 problem 时一并提出，避免 rebuttal 时被动）

| 风险 | 描述 | 缓解 |
|------|------|------|
| **R1**: $\Phi^{\!\top}\Phi$ 在边界处接近 0 | 数据保真步分母 $\mu + \Phi\Phi^{\!\top}$ 不稳 | `PhiPhiT.clamp(min=1e-6)` |
| **R2**: SWAP 频域参数 $\alpha, v_s, v_\lambda, t$ 训练不稳 | Softplus 约束正值 + 初始化（见 §A5） |
| **R3**: KGD Born 近似失效 | 限制 $m_0^2 \le 0.5$；只在 KG 实验启用 |
| **R4**: $\sigma$ 估计偏差导致 AdaSpec 误抑制信号 | $\sigma_{\text{gate}}$ 使用 sigmoid（软门控），不做硬阈值 |
| **R5**: 共享权重在长 stage 数下表达不足 | 提供 `share_weights=False` 选项；论文给出两组对比 |

---

## 8. Notation 速查

| 符号 | 含义 | 维度 |
|------|------|------|
| $g$ | 测量值 | $H \times W'$ |
| $\Phi$ | sensing matrix（已展开） | $HW' \times HW\Lambda$ |
| $M / \Phi$ | 物理 mask（与 $\Phi$ 同义，在网络层面常写作 $\Phi$） | $H \times W$ 或 $H \times W \times \Lambda$（broadcast） |
| $\Phi^*$ | 退化 mask（shift + compress + reverse）| $H \times W \times \Lambda$ |
| $\Delta \Phi$ | SEC 估计的 sensing error | $H \times W \times \Lambda$ |
| $w$ | DAG 输出的退化权重 | $H \times W \times \Lambda$，值域 (0,1) |
| $\sigma$ | NLE 输出的噪声水平 | scalar per sample |
| $u, v$ | 波场振幅 / 速度场 | $H \times W \times \Lambda$ |
| $\alpha, v_s, v_\lambda, t$ | SWAP 物理参数（标量） | scalar |
| $\beta_k, \rho_k$ | Nesterov 动量、GD 步长（per stage） | scalar |
| $f^k$ | 第 $k$ 个 stage 的重建结果 | $H \times W \times \Lambda$ |

---

## 9. Section-level 写作提示（论文行文骨架）

- **Section 1 Introduction** ← §1.3、§4 各取 2 句；以 Insight 1 收尾。
- **Section 2 Related Work** ← §3（gap）+ §6（取舍）。
- **Section 3.1 Problem Setup** ← §1.1、§1.2、§A1-A4。
- **Section 3.2 Preliminaries** ← §2.5、§2.6、§2.7 的事实表述（不展开证明，证明放 algorithm.md）。
- **Section 5 Limitations / Discussion** ← §7。

> 写作时记得保留 Observation/Insight 的编号锚点（A/B/C/…），便于在 ablation 和 figure caption 中反向引用。
