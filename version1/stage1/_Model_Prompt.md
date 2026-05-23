# WaveMST 架构说明

> Pipeline、实现细节、论文图提示词。

---

## 1. 整体 Pipeline

```
原始高光谱图像 GT [B, 28, 256, 256]
         │
         ▼
  ┌─────────────────────────────────────────┐
  │            CASSI 前向过程                │
  │  masked  = mask3d * GT                  │
  │  shifted = shift(masked, step=2)        │  [B, 28, 256, 310]
  │  meas    = sum(shifted, dim=1)          │  [B, 256, 310]
  │  H       = shift_back(meas/28*2)        │  [B, 28, 256, 256]
  └─────────────────────────────────────────┘
         │ input_meas H
         ▼
  ┌─────────────────────────────────────────┐
  │           WaveMST 重建网络               │
  │   embedding → Encoder → Bottleneck      │
  │             → Decoder → mapping         │
  │   全局残差: output = mapping(fea) + H   │
  └─────────────────────────────────────────┘
         │ pred [B, 28, 256, 256]
         ▼
  Loss = RMSE(pred, GT)
  评估: PSNR / SSIM / SAM（对测试集 10 个场景）
```

---

## 2. 模型架构

### 2.1 U-Net 骨架（所有模型共用）

```
Input [B,28,H,W]
    │
    ▼  Conv2d(28→dim, 3×3) + LeakyReLU
    │
    ├──[Stage 0]── Block×n₀ ──→ skip₀ [B, dim,   H,   W  ]
    │                 │ Conv2d(dim→dim×2, 4,stride=2)
    ├──[Stage 1]── Block×n₁ ──→ skip₁ [B, dim×2, H/2, W/2]
    │                 │ Conv2d(dim×2→dim×4, 4,stride=2)
    │
    ├──[Bottleneck]── Block×n₂       [B, dim×4, H/4, W/4]
    │
    ├──[Decode 1]── ConvT + Cat(skip₁) + Fuse + Block×n₁
    ├──[Decode 0]── ConvT + Cat(skip₀) + Fuse + Block×n₀
    │
    ▼  Conv2d(dim→28, 3×3)
    +  残差(Input)
Output [B,28,H,W]
```

**mask 在 U-Net 中的流动**：
- 初始 mask_spatial = input_mask[:, :, :, :H]（从 shifted mask 截取）
- 每经过一个 encoder stage：`mask = sigmoid(Conv2d(mask))`（下采样到同分辨率）
- decoder 复用对应 encoder 层保存的 mask（skip connection）

### 2.2 各模型的 Block 差异

| 模型 | Block 内部结构 |
|------|---------------|
| Model 0 (WaveMST_3D) | LN → **WPO3D** → Res → LN → FFN → Res |
| Model 1 (WaveMST_KG) | LN → **WPO3D + Born修正** → Res → LN → FFN → Res |
| Model 2 (WaveMST_Parallel) | LN → **[WPO3D ‖ S-MSA] → 门控融合** → Res → LN → FFN → Res |
| Model 3 (WaveMST_Mamba) | LN → **WPO2D** → Res → LN → **SSM** → Res → LN → FFN → Res |

---

## 3. WPO3D 核心实现细节

### 3.1 数学原理

3D 各向异性阻尼波动方程：

```
∂²u/∂t² + α·∂u/∂t = vs²(∂²u/∂x² + ∂²u/∂y²) + vλ²·∂²u/∂λ²
```

频域闭式解（对 [C, H, W] 三个维度做 FFT）：

```
û(ω,t) = e^(-αt/2) · [û₀·Cs(η,t) + (v̂₀ + α/2·û₀)·Sn(η,t)]

η = ω₀² - (α/2)²,   ω₀² = vs²(ωx²+ωy²) + vλ²·ωλ²

η > 0（欠阻尼）: Cs=cos(√η·t),  Sn=sin(√η·t)/√η
η < 0（过阻尼）: Cs=cosh(√|η|·t), Sn=sinh(√|η|·t)/√|η|
```

### 3.2 代码流程（WPO3D.forward）

```
x [B,C,H,W]
    │
    ├── mask_spatial → MaskGateA → gate = 0.1 + 0.9·mask
    │                              u0 = Phi(x) * gate    [B,C,H,W]
    │                              v0 = Psi(x) * gate    [B,C,H,W]
    │
    ├── rfftn(u0, dim=(-3,-2,-1)) → u0_fft [B,C,H,W//2+1]  (复数)
    ├── rfftn(v0, dim=(-3,-2,-1)) → v0_fft
    │
    ├── 频率网格: fc[C,1,1], fh[1,H,1], fw[1,1,W//2+1]
    │   omega_sq = (2π)² · (vs²·(fh²+fw²) + vλ²·fc²)
    │   eta = omega_sq - (α/2)²
    │   is_under = (eta >= 0)
    │   cs   = where(is_under, cos(√|η|·t), cosh(√|η|·t))
    │   sinc = where(is_under, sin/√η,       sinh/√|η|)
    │
    ├── out_fft = e^(-αt/2) · (u0_fft·cs + (v0_fft + α/2·u0_fft)·sinc)
    │
    ├── irfftn(out_fft, s=(C,H,W), dim=(-3,-2,-1)) → out [B,C,H,W]
    │
    └── LayerNorm(out) → out * SiLU(x) → Conv1×1 → output
```

### 3.3 FFT 维度约定

```
torch.fft.rfftn(x, dim=(-3,-2,-1))
  输入:  [B, C, H, W]
  输出:  [B, C, H, W//2+1]  (复数)
  dim=-3 → C（光谱/通道维）
  dim=-2 → H（空间 y）
  dim=-1 → W（空间 x，rfft 只返回一半）

频率网格:
  freq_c = fftfreq(C)    → [C]      对应通道/光谱频率
  freq_h = fftfreq(H)    → [H]      对应空间 y 频率
  freq_w = rfftfreq(W)   → [W//2+1] 对应空间 x 频率（单边）
```

### 3.4 NaN 防护（关键）

```python
# 绝对不能直接 sqrt(eta)，当 eta<0 时会 NaN
pos = eta.clamp(min=0)          # 仅取正值部分
neg = (-eta).clamp(min=0)       # 仅取负值部分（eta<0 的情况）
omega_d = torch.sqrt(pos + 1e-30)
gamma   = torch.sqrt(neg + 1e-30)
# 用 torch.where 分区合并
```

### 3.5 可学习参数初始化与约束

```python
self.alpha = nn.Parameter(torch.tensor(0.1))  # 阻尼
self.vs    = nn.Parameter(torch.tensor(1.0))  # 空间波速
self.vl    = nn.Parameter(torch.tensor(0.5))  # 光谱波速
self.t     = nn.Parameter(torch.tensor(1.0))  # 传播时间

# 所有参数用 softplus 激活保证正值
alpha_eff = F.softplus(self.alpha)
```

---

## 4. Mask 机制对比

| 方案 | 数学形式 | 物理含义 | 复杂度 | 代码位置 |
|------|---------|---------|--------|---------|
| A（默认） | u0 = Phi(x)·gate，gate=ε+(1-ε)M | 对应 CASSI 一次性振幅调制 | O(NlogN) | MaskGateA |
| B | 在主传播基础上叠加 M·S(x) 源项 | mask 持续注入光场 | O(1.5·NlogN) | MaskSourceB |
| D | 零阶WPO + Born一阶修正，质量场m²=(1-M) | Klein-Gordon 质量场，mask=0处惯性大 | O(2·NlogN) | MaskKleinGordonD |

---

## 5. CASSI shift/shift_back 说明

```
shift(mask3d*gt, step=2):
  波段 i 的像素列向右偏移 2·i 个像素
  [B, 28, 256, 256] → [B, 28, 256, 310]
  (310 = 256 + 27×2)

shift_back(meas, step=2):  [dataset.py 版本，用于 gen_meas]
  [B, 256, 310] → [B, 28, 256, 256]
  从第 2i 列开始截取 256 列

shift_back(mask, step=2):  [mst.py 版本，用于 MaskGuidedMechanism]
  [B, nC, H, W_shifted] → [B, nC, H, H]
  步长按分辨率缩放（下采样后 step 变小）
```

---

## 6. 模型图提示词（用于 AI 作图）

### 6.1 整体框架图（英文，Sora/Midjourney 风格）

```
A clean scientific architecture diagram for a deep learning paper.
Style: white background, flat design, no 3D effects, academic publication quality.
Content: Show a U-Net style encoder-decoder network for hyperspectral image reconstruction.
Left: input measurement [B,28,256,256] enters an embedding layer (Conv2d).
Encoder: 2 downsampling stages, each containing stacked WPO blocks (blue rectangles labeled "WPO3D Block").
Each block has: LayerNorm → Wave Propagation Operator → Residual Add → LayerNorm → FFN → Residual Add.
Between stages: stride-2 convolution for feature downsampling, and separate mask downsampling path shown as a thin orange arrow.
Bottleneck: 2 WPO3D blocks at 1/4 spatial resolution.
Decoder: 2 upsampling stages with skip connections shown as horizontal dashed arrows from encoder to decoder.
Right: output [B,28,256,256] after a final Conv2d + global residual connection.
Color coding: blue for WPO blocks, green for FFN, orange for mask path, gray for convolutions.
Add small math labels: "3D-FFT", "Wave Modulation", "3D-IFFT" inside the WPO3D block.
```

### 6.2 WPO3D 核心模块图

```
A detailed module diagram for a scientific paper figure.
White background, clean lines, academic style.
Show a single "WPO3D Block" as a vertical flow diagram:

Input feature x [B, C, H, W]
  ↓ LayerNorm
  ↓ Split into two paths:
    Left path: DWConv + Linear → u0 (initial field)
    Right path: DWConv + Linear → v0 (velocity field)
    Both paths multiplied by "Mask Gate = ε + (1-ε)·M" (shown as a mask icon)
  ↓ 3D rFFT on (C, H, W) dims
  ↓ "Wave Modulation" box containing:
      ω² = (2π)²[vs²(ωx²+ωy²) + vλ²ωλ²]
      η = ω² - (α/2)²
      Underdamped (η>0): cos/sin terms
      Overdamped (η<0): cosh/sinh terms
      decay = exp(-αt/2)
  ↓ 3D irFFT → out [B, C, H, W]
  ↓ LayerNorm → × SiLU(x) → Conv1×1
  ↓ Residual Add with original x
  ↓ LayerNorm → FFN (Conv1×1→DWConv→Conv1×1)
  ↓ Residual Add

Learnable parameters shown in a sidebar: α (damping), vs (spatial speed), vλ (spectral speed), t (time).
Use blue color for FFT/IFFT boxes, red for wave modulation, green for mask gate.
```

### 6.3 四种模型对比图

```
A 2×2 grid comparison figure for a deep learning paper.
White background, clean academic style.
Show 4 variants of the same block design, each in a separate quadrant:

Top-left (Model 0 - WaveMST_3D):
  [LN] → [3D WPO] → [+] → [LN] → [FFN] → [+]
  Label: "WaveMST-3D: Pure 3D Wave Propagation"

Top-right (Model 1 - WaveMST_KG):
  [LN] → [3D WPO] → [Born Correction] → [+] → [LN] → [FFN] → [+]
  Label: "WaveMST-KG: + Klein-Gordon Mass Field"

Bottom-left (Model 2 - WaveMST_Parallel):
  [LN] → [3D WPO] ─┐
  [LN] → [S-MSA]  ─┤ [Gate] → [+] → [LN] → [FFN] → [+]
  Label: "WaveMST-Parallel: Physics + Data-Driven"

Bottom-right (Model 3 - WaveMST_Mamba):
  [LN] → [2D WPO] → [+] → [LN] → [1D SSM] → [+] → [LN] → [FFN] → [+]
  Label: "WaveMST-Mamba: Spatial Wave + Spectral SSM"

Use consistent color coding: blue=WPO, purple=attention/SSM, green=FFN, orange=gate.
```

### 6.4 CASSI 物理过程图

```
A physics diagram illustrating the CASSI (Coded Aperture Snapshot Spectral Imager) process.
White background, clean scientific illustration style.

Show the light path from left to right:
1. 3D hyperspectral cube [H×W×28 wavelengths] — shown as stacked colored rectangles
2. Arrow pointing right: "Coded Aperture (Mask M)"
3. Binary mask pattern applied — each wavelength band gets spatially multiplied by M
4. Arrow: "Dispersive Element (Prism)"
5. Each wavelength band shifts horizontally by 2i pixels (show 3 colored bands offset)
6. Arrow: "2D Sensor Integration"
7. Final 2D measurement [256×310] — all bands summed onto one detector

Below the main diagram, show the reconstruction network taking the 2D measurement as input
and outputting the 3D hyperspectral cube.

Add mathematical annotations:
  y = Σᵢ mask(x,y)·f(x, y, λᵢ) where the sum is over all shifted bands
  H = shift_back(y/28·2)  ← network input initialization

Use wavelength-appropriate colors (violet to red) for the spectral bands.
```

---

## 7. 文件依赖关系

```
train.py / test.py
    ├── dataset.py        (shift, shift_back, gen_meas, load_*, shuffle_crop)
    ├── loss.py           (rmse_loss, torch_psnr, torch_ssim, torch_sam)
    └── wpo3d.py          (Model 0/1)
        ├── mask_ops.py   (MaskGateA, MaskSourceB, MaskKleinGordonD)
        ├── wpo_smsa.py   (Model 2) ← 依赖 wpo3d.py + mst.py
        └── wpo_mamba.py  (Model 3) ← 依赖 wpo3d.py
            └── mst.py    (MST baseline + MS_MSA 组件)

viz.py                    (独立，仅依赖 matplotlib/numpy)
dataset/mat2npy.py        (独立，仅依赖 scipy/numpy)
```
