# SMILE² 命名映射表

> **论文标题**：SMILE²: Spectral Modulated Imaging via Learned Estimation-Evolution  for Snapshot Compressive Reconstruction
>
> **SMILE²** = **S**pectral **M**odulated **I**maging via **L**earned **E**stimation-**E**volution

---

## 总体框架

| 层级 | 名称 | 缩写 | 对应 SMILE² |
|------|------|------|------------|
| **整体模型** | **SMILE²** | — | 全部 |

---

## Part I — 物理传播器

| 组件 | 名称 | 缩写 | 技术内容 |
|------|------|------|---------|
| **整体** | **Spectral WAve Propagator** | **SWAP** | 3D 波传播 + Mask 门控 + AdaSpec |
| 核心算子 | 3D Damped Wave Equation (闭式解) | — | rFFT → cos/sin 调制 → irFFT |
| Mask 融入 | Modulated Initialization | **MI** | gate = ε + (1−ε)Φ，对应 MaskGateA |
| 频带加权 | Adaptive Spectral Filtering | **AdaSpec** | W = σ((|û|²−σ²)/(|û|²+σ²+ε))，零参数 SNR 自适应 |
| KG 色散（可选） | Klein-Gordon Dispersion | **KGD** | k²(λ) = (2π/λ)² + Born 一阶修正，对应 MaskKleinGordonD |
| Swin 窗口（可选） | Windowed Wave Propagation | **W-SWAP** | 64×64 窗内传播 + shift window |
| Block 单元 | SWAP Block | — | LN → SWAP → Res → LN → FFN → Res |

---

## Part II — 退化估计与精化

| 组件 | 名称 | 缩写 | 技术内容 |
|------|------|------|---------|
| **整体** | **Learned Degradation Estimator** | **LDE** | 三合一退化估计 + 局部精化 |
| 子组件 1 | Sensing Error Correction | **SEC** | ΔΦ：残差学习修正 sensing matrix |
| 子组件 2 | Degradation-Aware Gating | **DAG** | deg_weight ∈ [0,1]：空间退化权重（Sigmoid） |
| 子组件 3 | Noise Level Estimator | **NLE** | σ > 0：噪声水平（Softplus），控制 SWAP 阻尼 |
| 子组件 4 | Local Refinement Block | **LRB** | DWConv 3×3 + GELU + Conv 1×1，补局部纹理 |

---

## Part III — 展开框架

| 组件 | 名称 | 缩写 | 技术内容 |
|------|------|------|---------|
| **整体** | **Accelerated Half-Quadratic Splitting** | **A-HQS** | Nesterov 动量加速展开 |
| 动量项 | Nesterov Momentum | — | f̂ = f^k + β_k(f^k − f^{k−1}) |
| GD step | Data Fidelity Step | — | z = f̂ + ρ·Φ_eff^T(g−Φ_eff·f̂)/(ΦΦ^T) |
| 步长预测 | Para Estimator | — | ρ_k = softplus(CNN(f)) |
| 多阶段损失 | Multi-Stage Loss | — | Σ w_k · RMSE(f^k, GT) |

---

## 完整信号流中的命名标注

```
输入: g (measurement), Φ (mask)

Stage k = 1 ... K:

  ┌─ LDE ────────────────────────────┐
  │  SEC:  ΔΦ = Conv(Φ)              │  ← Sensing Error Correction
  │  DAG:  w  = Sigmoid(Conv(Φ,Φ*))  │  ← Degradation-Aware Gating
  │  NLE:  σ  = Softplus(MLP(f))     │  ← Noise Level Estimator
  └──────────────────────────────────┘
                    │
                    ▼
  ┌─ A-HQS ─────────────────────────┐
  │  Momentum: f̂ = f + β(f − f_prev)│  ← Nesterov 动量
  │  GD step:  z = f̂ + ρ·Φ_eff^T(…) │  ← 数据保真（用 Φ+ΔΦ 修正）
  └──────────────────────────────────┘
                    │
                    ▼
  ┌─ DAG 净化 ──────────────────────┐
  │  z_clean = z · (1 + w)           │  ← 退化加权
  │  z_clean = LayerNorm(z_clean)    │  ← 归一化守门
  └──────────────────────────────────┘
                    │
                    ▼
  ┌─ SWAP ──────────────────────────┐
  │  MI:      u₀,v₀ = gate(z_clean) │  ← Modulated Initialization
  │  3D FFT → Wave Modulation       │  ← 阻尼波动方程闭式解
  │  AdaSpec: out *= W(ω,σ)         │  ← Adaptive Spectral Filtering
  │  [KGD]:   Born correction       │  ← Klein-Gordon Dispersion (可选)
  │  3D iFFT → SiLU gate → Conv     │
  │  U-Net: enc → bottleneck → dec  │  ← SWAP Block × N 组成
  └──────────────────────────────────┘
                    │
                    ▼
  ┌─ LRB ──────────────────────────┐
  │  f = f_wave + DWConv_FFN(f_wave)│  ← Local Refinement Block
  └──────────────────────────────────┘
                    │
                    ▼
  输出: f^{k+1} → out_list
```

---

## 论文行文示例

> We propose **SMILE²** (**S**pectral **M**odulated **I**maging via **L**earned **E**stimation-**E**volution), a physics-informed deep unfolding framework for snapshot compressive spectral imaging. SMILE² consists of three components:
>
> (1) **SWAP** (Spectral WAve Propagator): a 3D damped wave equation solver operating in the frequency domain, equipped with **Modulated Initialization (MI)** that encodes the CASSI mask into the wave field's initial conditions, and **AdaSpec** (Adaptive Spectral Filtering) that performs SNR-guided frequency band weighting at zero parameter cost.
>
> (2) **LDE** (Learned Degradation Estimator): a lightweight module (~5K parameters) that simultaneously estimates sensing errors (**SEC**), spatial degradation weights (**DAG**), and noise levels (**NLE**) to purify the initial field before wave propagation.
>
> (3) An **A-HQS** (Accelerated Half-Quadratic Splitting) iterative framework with Nesterov momentum that achieves $O(1/K^2)$ convergence, where each stage sequentially applies LDE → A-HQS → SWAP → **LRB** (Local Refinement Block).
>
> Optionally, SWAP can be augmented with **KGD** (Klein-Gordon Dispersion), which injects the physical wavenumber $k(\lambda) = 2\pi/\lambda$ as a hard spectral prior for improved spectral fidelity (SAM).

---

## 缩写速查表

| 缩写 | 全称 | 所属 |
|------|------|------|
| **SMILE²** | Spectral Modulated Imaging via Learned Estimation-Evolution | 整体 |
| **SWAP** | Spectral WAve Propagator | Part I 整体 |
| **MI** | Modulated Initialization | Part I · Mask 门控 |
| **AdaSpec** | Adaptive Spectral Filtering | Part I · 频带加权 |
| **KGD** | Klein-Gordon Dispersion | Part I · 色散（可选）|
| **W-SWAP** | Windowed SWAP | Part I · Swin 窗口（可选）|
| **LDE** | Learned Degradation Estimator | Part II 整体 |
| **SEC** | Sensing Error Correction | Part II · ΔΦ |
| **DAG** | Degradation-Aware Gating | Part II · deg_weight |
| **NLE** | Noise Level Estimator | Part II · σ |
| **LRB** | Local Refinement Block | Part II · 局部精化 |
| **A-HQS** | Accelerated Half-Quadratic Splitting | Part III |
| **E²** | Estimation-Evolution | SMILE² 的核心创新对 |

