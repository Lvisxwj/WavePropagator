# CLAUDE.md — SMILE² Codebase Quick Reference

> Auto-generated index for fast navigation when working on this project.

## 项目定位

CASSI（Coded Aperture Snapshot Spectral Imaging）高光谱图像重建任务的深度展开框架，论文代号
**SMILE²** = *Spectral Modulated Imaging via Learned Estimation-Evolution*。

主推模型架构：**Purify → Propagate → Refine**

- Part I — **SWAP**：3D 阻尼波动方程闭式解 + Mask 软门控 + AdaSpec（频带加权）
- Part II — **LDE**：三合一退化估计（SEC + DAG + NLE）+ LRB 局部精化
- Part III — **A-HQS**：Nesterov 动量 + 数据保真闭式解 + 多 stage 损失

## 目录约定

```
src/
├── version1/                            # 旧版（GAP unfolding），保留对照
├── version2/                            # 主推代码
│   ├── config.yaml                      # 训练超参 + 数据路径
│   ├── __init__.py                      # 模块开关
│   ├── train.py / test.py / dataset.py  # 训练/测试/数据
│   ├── loss.py                          # RMSE 损失 + PSNR/SSIM/SAM 指标
│   └── model/
│       ├── wpo3d.py        ← Part I  : WPO3D + Block + WaveMST_3D/_KG
│       ├── mask_ops.py     ← Part I  : MaskGateA (MI) + MaskKleinGordonD (KGD)
│       ├── degradation.py  ← Part II : DegradationEstimation (SEC+DAG+NLE) + construct_degraded_mask
│       ├── refinement.py   ← Part II : LocalRefinement (LRB)
│       ├── unfolding.py    ← Part III: WPO_Unfold (A-HQS / GAP 切换)
│       └── utils.py        ← Part III: shift_batch, mul_Phi_f, mul_PhiT_residual, ParaEstimator
├── paper/
│   ├── analysis/                        # 设计/推导 md（数学闭式解、决策、SOTA 对比）
│   ├── reference_paper_pdf/             # DPU / SSR / WaveFormer / Heat-former / Phy-CoSF 等
│   └── my work/                         # 论文 md 产出（problem / algorithm / architecture）
└── CLAUDE.md                            # 本文件
```

## 名称-代码对照（基准）

| 论文术语 | 代码位置 |
|---------|---------|
| **SWAP** | `model/wpo3d.py: WPO3D / WPO3DBlock / WaveMST_3D` |
| **MI** (Modulated Initialization) | `model/mask_ops.py: MaskGateA` |
| **AdaSpec** (Adaptive Spectral Filtering) | `model/wpo3d.py: WPO3D._apply_fbgw`，对应 `fbgw_mode='snr_adaptive'` |
| **KGD** (Klein-Gordon Dispersion) | `model/mask_ops.py: MaskKleinGordonD`, `model/wpo3d.py: WaveMST_KG` |
| **W-SWAP** (Windowed SWAP) | `model/wpo3d.py: WPO3D._swin_forward`，`use_swin=True` |
| **LDE** | `model/degradation.py: DegradationEstimation` |
| **SEC** (Sensing Error Correction) | `LDE.delta_phi → ΔΦ` |
| **DAG** (Degradation-Aware Gating) | `LDE.deg_weight → w` |
| **NLE** (Noise Level Estimator) | `LDE.sigma_est → σ` |
| **LRB** (Local Refinement Block) | `model/refinement.py: LocalRefinement` |
| **A-HQS** | `model/unfolding.py: WPO_Unfold` (when `use_ahqs=True`) |
| **Para Estimator** (ρ_k) | `model/utils.py: ParaEstimator` |
| **Nesterov momentum** (β_k) | `WPO_Unfold.betas` |

## 关键开关（version2/__init__.py）

```python
USE_KG               = False              # 'A' 或 'D' mask mode
WPO_FBGW_MODE        = 'snr_adaptive'     # 'none' / 'snr_adaptive' / 'learnable_band'
USE_SWIN_WPO         = False              # 64×64 窗内传播
USE_UNFOLDING        = True
USE_AHQS             = False              # A-HQS（动量+ΔΦ）或 GAP
NUM_STAGES           = 5
SHARE_STAGE_WEIGHTS  = True
MULTI_STAGE_LOSS     = True
```

## 单个 unfolding stage 数据流（version2 默认）

```text
输入: f^{k-1}, g, Φ, Φ*(预计算)
1. LDE(f, Φ, Φ*)  → ΔΦ, w, σ
2. [A-HQS] f̂ = f + β_k(f − f_prev)             # Nesterov 动量
3. GD  z = f̂ + ρ_k · Φ_eff^T(g − Φ_eff f̂) / (μ+ΦΦ^T)
                                                ↑  ρ_k = ParaEstimator(f̂)
                                                ↑  Φ_eff = Φ + ΔΦ
4. z_clean = z · (1 + w)                        # DAG 净化
5. f_wave = SWAP(z_clean, Φ, σ)                 # σ → α_eff = α + λ_σ σ
6. f      = f_wave + LRB(f_wave)
输出: f^k
```

主多 stage 损失：`L = Σ_k w_k · RMSE(f^k, GT)`，权重 `[..., 0.3, 0.5, 0.7, 1.0]`。

## 数学骨干

各向异性 3D 阻尼波动方程：

$$\partial_{tt}u + \alpha \partial_t u = v_s^2(\partial_{xx}+\partial_{yy})u + v_\lambda^2 \partial_{\lambda\lambda} u$$

频域闭式解（统一欠/过阻尼）：

$$\hat u(\boldsymbol\omega, t) = e^{-\alpha t/2}\left[\hat u_0 \cdot \mathrm{Cs}(\eta, t) + (\hat v_0 + \tfrac{\alpha}{2}\hat u_0)\cdot \mathrm{Sn}(\eta, t)\right]$$

其中 $\eta = \omega_0^2 - (\alpha/2)^2$，$\omega_0^2 = v_s^2(\omega_x^2 + \omega_y^2) + v_\lambda^2\omega_\lambda^2$，$\mathrm{Cs}/\mathrm{Sn}$ 是符号自适应的 cos/cosh、sin/sinh。

## 常用命令

| 任务 | 命令 |
|------|------|
| 语法检查 | `python -m py_compile version2/model/*.py` |
| Forward 自检（CUDA） | 参考 `version2/QUICKSTART.md` Step 2 |
| 训练 | `cd version2 && python train.py` |
| 测试 | 设置 `BEST_CKPT`, 然后 `python test.py` |

## 参考论文（paper/reference_paper_pdf）

- WaveFormer — Frequency-Time Decoupled Vision Modeling（路线 3 闭式解原型）
- Heat-former — Heat-equation-based modeling（阻尼对照）
- DPU CVPR 2024 — Dual Prior Unfolding（双 prior + Focused Attention）
- SSR CVPR 2024 — Spectral-Spatial Rectification（WSSA + ARB）
- Phy-CoSF — Physics-Guided Continuous Spectral Fields（A-HQS + 物理先验）
- MST CVPR 2022 — Mask-Guided Spectral-Wise Transformer（baseline）

## 关键 md（paper/）

| 文件 | 主题 |
|------|------|
| `analysis/1basicwpo.md` | 路线 3 闭式解推导 + Mask A/B/C/D 方案 + Transformer 处理路线 |
| `analysis/2deepunfolding.md` | DPU / SSR 拆解 + 与 SWAP 的差距来源 |
| `analysis/3New Framework Decisions.md` | LDE 设计、AdaSpec、A-HQS 决策 |
| `analysis/Thought Analysis Redesign.md` | 净化-传播-精化范式确立 |
| `my work/name_mapping.md` | **论文术语 ↔ 代码命名 权威映射表（基准）** |
| `my work/logic.md` | 写作任务清单（problem / algorithm / architecture） |
