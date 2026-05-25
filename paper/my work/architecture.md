# architecture.md — SMILE² 完整组件视图（绘图蓝本）

> 本文档是论文 **Architecture 图**的视觉规范与组件清单。所有命名严格遵循
> `name_mapping.md`，颜色与符号约定下文统一，便于一次性把论文里的总图、各
> Part 的局部图、消融图绘成一套视觉体系。
>
> 子图与 ONNX 描述文件放在 `paper/my work/components/`：
>
> ```
> components/
> ├── swap.py / swap.onnx.json
> ├── swap-MI.py / swap-MI.onnx.json
> ├── swap-AdaSpec.py / swap-AdaSpec.onnx.json
> ├── swap-KGD.py / swap-KGD.onnx.json
> ├── swap-WSWAP.py / swap-WSWAP.onnx.json
> ├── swap-Block.py / swap-Block.onnx.json
> ├── lde.py / lde.onnx.json
> ├── lde-SEC.py / lde-SEC.onnx.json
> ├── lde-DAG.py / lde-DAG.onnx.json
> ├── lde-NLE.py / lde-NLE.onnx.json
> ├── lde-LRB.py / lde-LRB.onnx.json
> ├── ahqs.py / ahqs.onnx.json
> ├── ahqs-ParaEstimator.py / ahqs-ParaEstimator.onnx.json
> ├── ahqs-Momentum.py / ahqs-Momentum.onnx.json
> └── ahqs-MultiStageLoss.py / ahqs-MultiStageLoss.onnx.json
> ```
>
> 每个 `.py` 文件按 `nn.Module` 形式表达组件结构（便于读者一眼看出 IO）；
> 对应的 `.onnx.json` 文件以 ONNX 图（nodes / edges / metadata）格式记录
> 每个算子的精确接线、颜色、所属 Part，便于直接喂给 Netron-like 渲染或
> 论文 figure 工具（draw.io、TikZ、Manim 等）。

---

## 1. 视觉系统

### 1.1 调色板（最终选定）

| 用途 | 颜色 (HEX) | 含义 |
|------|-----------|------|
| **Part I 背景**：SWAP（物理传播器） | `#f3f2f7` | 极浅紫灰，柔和的“频域 / 物理”体感 |
| **Part II 背景**：LDE（估计 / 退化） | `#fff7e6` | 暖米黄，区分“分析 / 估计”流派 |
| **Part III 背景**：A-HQS（展开框架） | `#e6f1ff` | 冷淡蓝，对应“迭代算法”视觉 |
| **强调色（主标题、关键箭头）** | `#3a155c` | 用户指定，深紫，对比强 |
| **次强调色（次箭头、虚线）** | `#7a4dba` | `#3a155c` 的浅化版 |
| **黑色文本 / 边框** | `#1a1a1a` | 非纯黑，避免刺眼 |
| **辅助网格、注释** | `#9a9aa3` | 中性灰 |

### 1.2 算子色（跨 Part 的“原子”操作统一颜色）

> 用同色卡片表达同类算子，读者跨 Part 比较时可秒认。

| 算子 | 颜色 (HEX) | 备注 |
|------|-----------|------|
| `Conv 1×1` | `#5dade2` | 浅蓝（线性混合） |
| `Conv 3×3` (含 DW3×3) | `#48c9b0` | 蓝绿（局部聚合） |
| `LayerNorm / LN2d` | `#f5b041` | 橙黄（归一化） |
| `Softplus` | `#a569bd` | 紫色（保正） |
| `Sigmoid / σ_gate` | `#ec7063` | 红粉（门控） |
| `GELU / SiLU / LeakyReLU` | `#f8c471` | 米黄（激活） |
| `FFT / iFFT (3D rFFT)` | `#3a155c` | 深紫（频域核心） |
| `Wave Modulate (Cs / Sn)` | `#7a4dba` | 中紫（核心物理调制） |
| `Element-wise × / +` | `#85929e` | 浅灰（运算） |
| `Concat` | `#dc7633` | 橙（拼接） |
| `AvgPool / GAP` | `#cd6155` | 深红（统计） |
| `Conv4×4 stride 2 (Down) / Up` | `#1abc9c` | 蓝绿（U-Net 采样） |

### 1.3 边的样式约定

- **实线 + 箭头**：张量正向流动；
- **粗实线**：跨 Part 主流（如 LDE → SWAP）；
- **虚线**：参数化耦合（$\sigma \dashrightarrow \alpha_{\mathrm{eff}}$、$\Delta\Phi \dashrightarrow \Phi_{\mathrm{eff}}$）；
- **点线**：可选分支（KGD、W-SWAP）；
- **波浪线**：周期 / 迭代回路（stage $k \to k+1$）。

### 1.4 Part 框图编排建议

```
┌──────────────────────────────────────────────────────────────┐
│  Stage k                                                    │
│  ┌──────────┐    σ ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐
│  │  LDE     │    │                                        │   │
│  │ (Part II)│────w────► ▶  z*(1+w) ─────►  SWAP (Part I)  │   │
│  │  ΔΦ──╮   │                              (受 σ 调阻尼)   │   │
│  │      │   │                                        │   │   │
│  └──────│───┘                                        ▼   │   │
│         ▼                                            ┌──────┐│
│   Φ_eff ──► A-HQS GD step (Part III) ──► z ──────►  │ LRB ││
│         (Nesterov momentum β_k)                      └──────┘│
│                                                       │      │
│                                                       ▼      │
│                                            ──── 输出 f^k ────│
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 总体三段视图

### 2.1 SMILE² 全局信号流（用于论文 Figure 1）

```
g (B,1,H,W')      Φ (B,Λ,H,W)
   │                  │
   └──────┬───────────┘
          ▼
    [Init Conv]   ←  initial_conv(2Λ→Λ, 1×1)
          │
          ▼   f^0
   ┌──────────────────── repeat K stages (default K=5) ────────────────────┐
   │                                                                       │
   │   ┌─────────────────────────────┐                                     │
   │   │     Part II : LDE           │                                     │
   │   │     SEC   ΔΦ                │                                     │
   │   │     DAG   w                 │                                     │
   │   │     NLE   σ                 │                                     │
   │   └────┬────────────┬────────┬──┘                                     │
   │  ΔΦ    │    w       │   σ    │                                        │
   │        ▼            │        │                                        │
   │   ┌─ Part III ───┐  │        │                                        │
   │   │ Momentum β_k │  │        │                                        │
   │   │ GD step (Φ_eff)│ │        │                                        │
   │   │ rho_k (Para) │  │        │                                        │
   │   └────┬─────────┘  │        │                                        │
   │       z             │        │                                        │
   │       ▼             │        │                                        │
   │      z·(1+w) ◄──────┘        │                                        │
   │       │                      │                                        │
   │       ▼                      │                                        │
   │   ┌─ Part I : SWAP ─────────────────────────────┐                     │
   │   │  MI   →  u0,v0                              │                     │
   │   │  3D rFFT  →  ŷ0,ŷ0                          │                     │
   │   │  Wave Modulate (α_eff ← α + λ_σ σ)          │←── σ ───────────────│
   │   │  AdaSpec (1.16)                             │                     │
   │   │  [opt] KGD Born correction                  │                     │
   │   │  [opt] W-SWAP windowed                      │                     │
   │   │  3D irFFT  →  f_wave                        │                     │
   │   └────┬────────────────────────────────────────┘                     │
   │        │                                                              │
   │        ▼                                                              │
   │   LRB  (DWConv FFN)                                                   │
   │        │                                                              │
   │        ▼                                                              │
   │   f^k  ──────►  L_k  (RMSE × w_k)  → multi-stage loss                │
   │                                                                       │
   └───────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                           f^K (final)
```

---

## 3. Part I — SWAP（Spectral WAve Propagator）

### 3.1 顶层 SWAP（`components/swap.py`、`swap.onnx.json`）

输入：
- `x` ∈ `[B, Λ, H, W]`（z_clean）
- `mask_spatial` ∈ `[B, Λ, H, W]`
- `sigma` ∈ `[B, 1, 1, 1]`（可空）

输出：
- `f_wave` ∈ `[B, Λ, H, W]`

主流：

```
x ─► Embedding(Conv3×3, Λ→Λ) ─► LeakyReLU ─► Encoder×S
                                       └─►   (跨 stage skip)
                                       ▼
                                  Bottleneck
                                       ▼
                                   Decoder×S
                                       ▼
                                Mapping(Conv3×3, Λ→Λ)
                                       │
                                       ▼
                                 + x  (global residual)
                                       │
                                       ▼
                                  f_wave
```

> Encoder/Decoder/Bottleneck 内部均由 **SWAP-Block** 组成（见 §3.7）。

### 3.2 MI — Modulated Initialization（`components/swap-MI.py`）

```
x ──► φ : DW3×3 + Conv1×1 ──► (multiply gate) ──► u0
mask ─┐                                          ▲
      └─► gate = ε + (1-ε)·mask  (ε=0.1) ────────┘
x ──► ψ : DW3×3 + Conv1×1 ──► (multiply gate) ──► v0
```

输出：`u0, v0 ∈ [B, Λ, H, W]`。颜色：Conv 系列遵循 §1.2；`gate` 用 `Sigmoid` 同色或独立浅蓝。

### 3.3 AdaSpec — Adaptive Spectral Filtering（`components/swap-AdaSpec.py`）

零参数路径（默认）：

```
û0 ─► |û0|²  ─┐
              ├─► (|û0|² − σ²) / (|û0|² + σ² + ε)  ─► sigmoid ─► W
σ²  ─────────┘
out_fft × W
```

可学版（`fbgw_mode='learnable_band'`）：

```
ω ─► |ω| ─► quantize to K bands ─► softplus(θ_k) lookup ─► W
out_fft × W
```

### 3.4 KGD — Klein-Gordon Dispersion（`components/swap-KGD.py`）

```
mask ─► (1 - mask) · m0² ──────► m²
u^(0) (after wave modulate, irfft) ──► · ──► -m²·u^(0) ─► FFT3D
                                        │
                                        ▼
                                Green(ω,t) = Sn(η,t)·e^{-αt/2}
                                        │
                                        ▼
                                       ×
                                        │
                                        ▼
                                  iFFT3D ─► u^(1)
u^(0) ──► + w_KG · u^(1) ─► f_wave (corrected)
```

### 3.5 W-SWAP — Windowed SWAP（可选，`components/swap-WSWAP.py`）

```
x, mask ──► partition into MxM windows ──► (per window) Global SWAP
                                                    │
                                                    ▼
                                       reassemble + optional shift
                                                    │
                                                    ▼
                                                  f_wave
```

### 3.6 Wave Modulate（核心算子，**所有 SWAP 实例共享**）

数学：公式 (1.9)。绘图建议把它做成一个高亮卡片，作为整张 Figure 的视觉重心：

```
û0 ──┐
ŷ0 ──┤                  
α,vs,vl,t,σ ──► [Compute Cs(η,t), Sn(η,t), decay] ──► out_fft
                       (α_eff = α + λ_σ·σ)
```

### 3.7 SWAP Block（`components/swap-Block.py`）

```
x ─► LN ─► WPO3D(SWAP) ──┐
                          + ──► LN ─► FFN ──┐
x ─────────────────────┘                     + ──► out
                                  x ─────────┘
```

`FFN`：`Conv1×1(Λ→4Λ) → GELU → DW3×3 → GELU → Conv1×1(4Λ→Λ)`。

---

## 4. Part II — LDE（Learned Degradation Estimator）

### 4.1 顶层 LDE（`components/lde.py`）

输入：`f, Φ, Φ* ∈ [B, Λ, H, W]`；输出：`ΔΦ, w ∈ [B, Λ, H, W], σ ∈ [B, 1, 1, 1]`。

```
Φ ─► SEC ─► ΔΦ
Φ ─┐
   ├─► Concat ─► DAG ─► w
Φ*─┘
f ─► NLE ─► σ
```

### 4.2 SEC — Sensing Error Correction（`components/lde-SEC.py`）

```
Φ ─► Conv1×1 ─► LeakyReLU ─► Conv1×1 ─► ΔΦ
```

### 4.3 DAG — Degradation-Aware Gating（`components/lde-DAG.py`）

```
[Φ ‖ Φ*] (2Λ) ─► Conv1×1 ─► LeakyReLU ─► Conv1×1 ─► Sigmoid ─► w (Λ)
```

### 4.4 NLE — Noise Level Estimator（`components/lde-NLE.py`）

```
f ─► GAP (1x1) ─► Flatten ─► Linear(Λ→h) ─► ReLU ─► Linear(h→1) ─► Softplus ─► σ
```

### 4.5 LRB — Local Refinement Block（`components/lde-LRB.py`）

> LRB 隶属 Part II（精化模块），但部署在 SWAP 之后，因此画图时常和 SWAP 同框；为避免歧义，仍归类 Part II。

```
x ─► Conv1×1(Λ→2Λ) ─► GELU ─► DW3×3(2Λ→2Λ) ─► GELU ─► Conv1×1(2Λ→Λ) ─► out
```

### 4.6 退化 mask 构造（`construct_degraded_mask`）

```
Φ ─► (shift each band by c·step) ─► Φ_shift ─► sum along Λ ─► Φ_comp ─► reverse (broadcast back) ─► 2·Φ_comp / Λ ─► Φ*
```

预计算（每个 batch 仅 1 次）。在视觉上常作为 LDE 输入旁的“辅助框”。

---

## 5. Part III — A-HQS（Accelerated Half-Quadratic Splitting）

### 5.1 顶层 A-HQS（`components/ahqs.py`）

输入：`g, Φ, Φ*, f^{k-1}, f^{k-2}, ΔΦ, w, σ`；输出：`f^k`。

```
f^{k-1}, f^{k-2}  ─► Momentum (β_k) ─► f̂
ΔΦ, Φ ──────────► Φ_eff = Φ + ΔΦ
f̂ ─► ParaEstimator (ρ_k) ─► ρ_k
Φ_eff, g, f̂, ρ_k ─► GD step (closed form, eq. 1.35) ─► z
z, w ─► z·(1+w) ─► z_clean
z_clean, Φ, σ ──► SWAP (Part I) ─► f_wave
f_wave ─► LRB ─► f_local
f_wave + f_local ─► f^k
```

### 5.2 ParaEstimator（`components/ahqs-ParaEstimator.py`）

```
f ─► Conv1×1(Λ→32) ─► ReLU ─► GAP ─► Conv1×1×2(MLP) ─► ReLU ─► Conv1×1 ─► + bias ─► Softplus ─► ρ_k
```

### 5.3 Nesterov Momentum（`components/ahqs-Momentum.py`）

```
f^{k-1}, f^{k-2} ─► (f^{k-1} - f^{k-2}) ─► × β_k ─► + f^{k-1} ─► f̂
β_k ← sigmoid(θ_k)
```

### 5.4 多 stage 损失（`components/ahqs-MultiStageLoss.py`）

```
[f^1, f^2, ..., f^K], f_GT
   for k in 1..K:
       L_k = RMSE(f^k, f_GT)
   L = Σ w_k · L_k    # w_K=1.0, w_{K-1}=0.7, w_{K-2}=0.5, w_{K-3}=0.3
```

---

## 6. 跨 Part 交叉关系（绘图时务必显式画出）

| 关系 | 起点 | 终点 | 含义 | 颜色 | 线型 |
|------|------|------|------|------|------|
| σ → α_eff | NLE | SWAP Wave-Modulate | 噪声感知阻尼 (1.36) | `#a569bd` | 虚线 |
| ΔΦ → Φ_eff | SEC | GD step | sensing 修正 (1.35) | `#a569bd` | 虚线 |
| w → z_clean | DAG | A-HQS Purify | 退化加权 (1.36) | `#a569bd` | 虚线 |
| Φ → MI gate | Init | SWAP MI | mask 软门控 (1.12) | `#7a4dba` | 实线 |
| 多 stage skip | f^{k-1} | f^k Momentum | Nesterov 动量 (1.33) | `#3a155c` | 波浪线 |
| Multi-stage loss | f^1..K | Loss | 加权损失 (1.38) | `#1a1a1a` | 实线 |
| `construct_degraded_mask` | Φ | Φ* | LDE 辅助输入 | `#9a9aa3` | 实线 |
| LRB ↔ SWAP | SWAP | LRB | 残差精化 (1.31) | `#1a1a1a` | 实线 |

---

## 7. 绘图建议（实操层）

1. **三 Part 用三色背景框**（见 §1.1），每个组件外用 1.5 px 黑色边框；
2. **算子用统一原子色 + 圆角矩形**（§1.2），避免“每个 layernorm 颜色不同”的视觉混乱；
3. **公式编号挂角标**：每个组件右上角放公式编号（如 SWAP 右上角写 (1.9)，AdaSpec (1.16) 等）；
4. **跨 Part 信号用 §6 表格里的虚/实/波浪线**；
5. **可选分支（KGD / W-SWAP）画在 SWAP 顶层框外侧**，点线连入；
6. **multi-stage 视图用横向折叠（一行画两 stage，省略其余）**，避免论文一页画不下；
7. **每张图右下角放 legend**：列出 §1.2 用到的原子色对应算子。

---

## 8. 与 name_mapping.md 的覆盖核对

| 论文术语 | 是否覆盖 | 文件 |
|---------|---------|------|
| **SMILE²** | ✓（§2.1） | `architecture.md` |
| **SWAP** | ✓ | `swap.py / swap.onnx.json` |
| **MI** | ✓ | `swap-MI.*` |
| **AdaSpec** | ✓ | `swap-AdaSpec.*` |
| **KGD** | ✓ | `swap-KGD.*` |
| **W-SWAP** | ✓ | `swap-WSWAP.*` |
| **SWAP Block** | ✓ | `swap-Block.*` |
| **LDE** | ✓ | `lde.*` |
| **SEC** | ✓ | `lde-SEC.*` |
| **DAG** | ✓ | `lde-DAG.*` |
| **NLE** | ✓ | `lde-NLE.*` |
| **LRB** | ✓ | `lde-LRB.*` |
| **A-HQS** | ✓ | `ahqs.*` |
| **Nesterov Momentum** | ✓ | `ahqs-Momentum.*` |
| **Data Fidelity Step (GD step)** | ✓ | 含在 `ahqs.*` |
| **Para Estimator (ρ_k)** | ✓ | `ahqs-ParaEstimator.*` |
| **Multi-Stage Loss** | ✓ | `ahqs-MultiStageLoss.*` |
| **E²（命名概念）** | ✓ | 通过 σ→α_eff、ΔΦ→Φ_eff 双虚线呈现 |

> 如果在绘图阶段发现遗漏的概念（例如 GD step 内 `mul_Phi_f`/`mul_PhiT_residual`），按 `ahqs.py` 内的子模块同色处理即可，不必单独建文件。

---

## 9. 后续可扩展

- 若新增 `swap-FlipAdaSpec` 或别名变体，请同步在 `components/` 创建对应 `.py + .onnx.json`，并在 §8 表格补齐核对；
- 若实验淘汰 KGD / W-SWAP / `learnable_band`，**保留文件**但在图中以浅灰打 X，便于论文 ablation 自动引用。
