# WaveMST 技术交接文档 — Claude Code 实现指南

> **目的**：本文档是给 Claude Code 的详细技术交接，包含论文思路、四种模型架构设计、代码目录规划、每个文件的实现要点、训练/测试 pipeline。Claude Code 应基于本文档 + 参考代码库完成全部实现。

---

## 0. 项目概述

### 0.1 一句话总结

将 WaveFormer（AAAI 2026）的波动方程传播算子 (WPO) 迁移到高光谱图像 (HSI) 的 CASSI 光谱压缩重建任务中，替代/增强 MST（CVPR 2022）的 Spectral-wise Self-Attention，引入物理约束的空-谱信息传播。 参照WaveMST_Analysis_Claude_Code.md

### 0.2 参考代码位置

| 参考项 | 路径 | 关注点 |
|-------|------|--------|
| MST 仿真训练 | `MST/simulation/train_code/` | train.py, utils.py, architecture/MST.py — U-Net 结构、mask 机制、CASSI 仿真、S-MSA |
| MST 仿真测试 | `MST/simulation/test_code/` | test.py, Quality_Metrics/ |
| WaveFormer WPO | `WaveFormer/WaveFormer/WaveFormer.py` | `Wave2D` 类 (L139-254)：DCT/IDCT、波速 `self.c`、阻尼 `self.alpha`、cos/sin 项调制 |
| 用户已有代码 | `vegetation_index/` | 简化版 MST pipeline，dataset 组织，可读性更好 |

### 0.3 数据集

```
dataset/
├── CAVE_1024_28/          # 训练集：scene1.npy ~ scene205.npy（或更多到242）
│                          # 每个 npy: shape (1024, 1024, 28), float32, 值域 [0,1]
└── TSA_simu_data/
    ├── mask.npy           # 仿真 mask: shape (256, 256), binary 0/1
    └── Truth/             # 测试集：scene01.npy ~ scene10.npy
                           # 每个 npy: shape (256, 256, 28), float32, 值域 [0,1]
```

**注意**：MST 原始数据是 `.mat` 格式，我们统一用 `.npy`。我已经转换好了。

---

## 1. 核心论文思路

### 1.1 问题

MST 用 Spectral-wise Multi-head Self-Attention (S-MSA) 沿光谱维度建模长距离依赖。但：
- **空间维度仅靠卷积**，缺乏全局传播
- **无物理先验约束**，纯数据驱动
- S-MSA 的 $O(N^2)$ 计算（虽然 N=28 波段不大，但空间维度上仍然受限）

### 1.2 我们的解法

用阻尼波动方程的闭式解构造 **Wave Propagation Operator (WPO)**，在空间/光谱维度做物理约束的全局传播，替代或增强 S-MSA。

### 1.3 核心公式

**3D 各向异性阻尼波动方程**：

$$\partial_{tt} u + \alpha \partial_t u = v_s^2(\partial_{xx}u + \partial_{yy}u) + v_\lambda^2 \partial_{\lambda\lambda}u$$

**频域闭式解**（这是 WPO 的数学基础）：

$$\hat{u}(\omega, t) = e^{-\alpha t/2}\left[\hat{u}_0 \cos(\omega_d t) + \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\omega_d}\sin(\omega_d t)\right]$$

其中阻尼频率：

$$\omega_d = \sqrt{v_s^2(\omega_x^2+\omega_y^2) + v_\lambda^2\omega_\lambda^2 - (\alpha/2)^2}$$

**实现步骤**：FFT → 频域乘以 cos/sin 调制系数 → IFFT。

**关键**：$\omega_d^2$ 可能为负（过阻尼区），此时 cos → cosh, sin → sinh。代码中必须用 `torch.where` 分情况处理，否则 `torch.sqrt` 会产生 NaN。

---

## 2. 四种模型架构

按优先级排列，`train.py` 中的模型索引从 0 开始。

### 2.1 Model 0（主推）：3D-WPO-Pure — 纯 3D WPO + Mask 初始门控

**思路**：完全舍弃 Transformer 的 S-MSA，用 3D WPO 同时处理空间+光谱的全局传播。Mask 作为初始振幅软门控。创新最大。

**架构（U-Net 风格，参照 MST）**：

```
Input: input_meas [B, 28, 256, 256]   (CASSI 初始化后的 H 或 HM)
Mask:  mask3d     [B, 28, 256, 256]   (3D shifted mask)

Embedding: Conv2d(28, dim, 3, 1, 1)  →  fea [B, dim, 256, 256]

Encoder Stage i (i=0,1,...,stage-1):
  ┌─ WPO3D_Block × num_blocks[i]:
  │    ├─ LN
  │    ├─ MaskGatedWPO3D(fea, mask_level_i)
  │    │    内部: soft_gate = eps + (1-eps)*mask_spatial
  │    │    u0 = Phi(fea) * soft_gate   (per-band spatial gating)
  │    │    v0 = Psi(fea) * soft_gate   (velocity encoder)
  │    │    3D FFT → 频域调制(cos/sin, learnable alpha,vs,vl,t) → 3D IFFT
  │    │    out_norm → SiLU gate → linear
  │    ├─ Residual add
  │    ├─ LN → FFN (Conv1x1 → GELU → DWConv3x3 → GELU → Conv1x1) → Residual
  │    └─ 
  ├─ fea_encoder[i] = fea   (skip connection)
  ├─ fea = FeaDownSample(fea)     Conv2d(dim_i, dim_i*2, 4, 2, 1)
  └─ mask = MaskDownSample(mask)  Conv2d(dim_i, dim_i*2, 4, 2, 1)

Bottleneck: WPO3D_Block × num_blocks[-1]

Decoder Stage i (i=0,1,...,stage-1):
  ├─ fea = FeaUpSample(fea)          ConvTranspose2d
  ├─ fea = Fusion(cat(fea, skip[i]))  Conv2d(dim*2, dim, 1, 1)
  ├─ mask = masks[i]                   (从 encoder 保存的)
  └─ WPO3D_Block × num_blocks[stage-1-i]

Mapping: Conv2d(dim, 28, 3, 1, 1)
Output: mapping(fea) + x   (全局残差)
```

**WPO3D 核心实现要点**（参照 WaveFormer 的 `Wave2D` 类，需要扩展为 3D）：

```
class WPO3D(nn.Module):
    # 可学习参数（每层独立）
    self.alpha = nn.Parameter(torch.tensor(0.1))   # 阻尼
    self.vs    = nn.Parameter(torch.tensor(1.0))    # 空间波速
    self.vl    = nn.Parameter(torch.tensor(1.0))    # 光谱波速
    self.t     = nn.Parameter(torch.tensor(1.0))    # 传播时间
    
    # 语义编码器
    self.phi = DWConv + Linear (生成 u0)
    self.psi = DWConv + Linear (生成 v0, velocity field)
    
    def forward(self, x, mask_spatial):
        # x: [B, C, H, W],  C=dim (与 28 波段通过 embedding 映射)
        # mask_spatial: [B, C, H, W]
        
        # Step 1: mask 软门控
        gate = eps + (1 - eps) * mask_spatial  
        
        # Step 2: 编码初始场
        u0 = self.phi(x) * gate    # [B, C, H, W]
        v0 = self.psi(x) * gate    # [B, C, H, W]
        
        # Step 3: 3D FFT  
        # 注意：C 维度对应光谱（dim 通道被视为光谱维度）
        # 对 (C, H, W) 三个维度做 FFT
        u0_fft = torch.fft.rfftn(u0, dim=(-3, -2, -1))
        v0_fft = torch.fft.rfftn(v0, dim=(-3, -2, -1))
        
        # Step 4: 构建频率网格
        freq_h = torch.fft.fftfreq(H)              # spatial freq y
        freq_w = torch.fft.rfftfreq(W)              # spatial freq x (rfft)
        freq_c = torch.fft.fftfreq(C)               # spectral freq
        # 广播为 3D 网格
        omega_sq = vs^2*(freq_h^2 + freq_w^2) + vl^2*freq_c^2
        
        # Step 5: 阻尼频率（处理过阻尼！）
        eta = omega_sq - (alpha/2)^2
        omega_d = torch.sqrt(torch.clamp(eta, min=0))
        gamma   = torch.sqrt(torch.clamp(-eta, min=0))
        is_underdamped = (eta > 0)
        
        # Step 6: 闭式解调制
        decay = torch.exp(-alpha * t / 2)
        
        cos_term = torch.where(is_underdamped,
                               torch.cos(omega_d * t),
                               torch.cosh(gamma * t))
        sinc_term = torch.where(is_underdamped,
                                torch.sin(omega_d * t) / (omega_d + 1e-8),
                                torch.sinh(gamma * t) / (gamma + 1e-8))
        
        out_fft = decay * (u0_fft * cos_term + 
                           (v0_fft + alpha/2 * u0_fft) * sinc_term)
        
        # Step 7: 3D IFFT
        out = torch.fft.irfftn(out_fft, s=(C, H, W), dim=(-3, -2, -1))
        
        # Step 8: 输出投影
        out = LayerNorm(out) → Linear → SiLU gate → Linear
        return out
```

**关于 C 维度与光谱维度的对应**：MST 中 `dim=28`（基础通道数等于波段数）。每个 encoder stage 通道数翻倍（28→56→112），此时"光谱传播"作用在通道维度上。这在物理上对应"在特征空间中的光谱传播"而非原始波段传播。这是可以接受的——WaveFormer 原论文也是在通道维度上做 WPO。

### 2.2 Model 1（激进高回报）：3D-WPO-KG — 3D WPO + Klein-Gordon Born 近似 + Mask 质量场

**思路**：在 Model 0 基础上，用 Klein-Gordon 方程替代普通波动方程，mask 直接进入物理方程核心（作为质量场），创新最大。

**与 Model 0 的唯一区别**：WPO3D 核心中增加 Born 一阶修正。

**Born 近似实现**：

```python
def forward_kg(self, x, mask_spatial):
    # --- 零阶解 u0 (与普通 WPO 相同) ---
    u0_out = wpo3d_standard_forward(x, mask_spatial)  # 就是 Model 0 的 WPO
    
    # --- 质量场 ---
    m_sq = self.m0_sq * (1 - mask_spatial)  # mask=0 处质量大
    # m0_sq 是可学习标量，限制范围 [0, 0.5]
    
    # --- Born 一阶修正 ---
    # 源项 = -m^2 * u0_out（空间域乘法）
    source = -m_sq * u0_out  # [B, C, H, W]
    
    # 对源项做 Duhamel 积分的离散近似
    # 简化：只用当前 t 的 Green 函数卷积（单步近似）
    source_fft = torch.fft.rfftn(source, dim=(-3,-2,-1))
    
    # Green 函数 = sinc_term * decay (已在零阶中计算)
    correction_fft = source_fft * sinc_term * decay  # 复用零阶的频域量
    correction = torch.fft.irfftn(correction_fft, s=(C,H,W), dim=(-3,-2,-1))
    
    # 总输出 = 零阶 + 一阶修正
    out = u0_out + self.kg_weight * correction  # kg_weight 初始化 0.1
    return out
```

**风险提示**：Born 近似要求 $m_0^2 t / \omega_0^2 \ll 1$，需要对 `self.m0_sq` 做 clamp。如果训练不稳定，可以在前 20 epoch 冻结 KG 项（`self.kg_weight=0`）。

### 2.3 Model 2（最稳妥）：3D-WPO-SMSA — 3D WPO 并联 S-MSA + Mask 初始门控

**思路**：保留 MST 的 S-MSA 作为光谱建模分支，同时并联 3D WPO 作为物理传播分支。两者融合。风险最低。

**架构修改（仅 Block 内部不同）**：

```
WPO_SMSA_Block:
  ├─ LN
  ├─ Branch 1: WPO3D(fea, mask)        → out_wpo  [B, dim, H, W]
  ├─ Branch 2: MS_MSA(fea, mask)        → out_smsa [B, dim, H, W]  
  │   (完全复用 MST 的 MS_MSA 类，包括 MaskGuidedMechanism)
  ├─ Fusion: Linear(cat(out_wpo, out_smsa)) → [B, dim, H, W]
  │   或者: gate * out_wpo + (1-gate) * out_smsa
  ├─ Residual add
  ├─ LN → FFN → Residual
  └─ 
```

**MS_MSA 直接从 MST 复制**（`MST/simulation/train_code/architecture/MST.py` L106-162）。注意它需要 shifted mask `[1, 28, 256, 310]`。mask 下采样时也需要对应处理。

**Fusion 方式选择**（提供两种，在 config 中切换）：
- `fusion='add'`：简单相加 `out = out_wpo + out_smsa`
- `fusion='gate'`：学习门控 `g = sigmoid(Linear(cat(wpo, smsa)))`, `out = g*wpo + (1-g)*smsa`

### 2.4 Model 3（探索性）：2D-WPO + 1D-Mamba — 空间 WPO + 光谱 Mamba + Mask 初始门控

**思路**：空间用 2D WPO（物理传播），光谱用 1D Mamba/SSM（线性复杂度序列建模）。空谱解耦但各自用最佳工具。

**架构**：

```
WPO_Mamba_Block:
  ├─ LN
  ├─ 空间 2D WPO:
  │    对每个通道独立做 2D FFT 波传播
  │    （完全同 WaveFormer 原版 Wave2D，加上 mask 门控）
  ├─ 光谱 1D Mamba:
  │    把 [B, C, H, W] reshape → [B*H*W, C]
  │    沿 C 维度做 1D SSM/Mamba
  │    （需要 mamba_ssm 包，或简化版 S4）
  ├─ 两者 output add → Residual
  ├─ LN → FFN → Residual
  └─
```

**Mamba 依赖**：需要 `pip install mamba-ssm`。如果服务器安装困难，可用简化版 S4（纯 PyTorch 实现的 1D SSM）。

**2D WPO 参照**：直接用 WaveFormer 的 `Wave2D` 类（L139-254），但需要：
1. 去掉 `torch_dct` 依赖，改用 `torch.fft.rfft2` / `torch.fft.irfft2`（原版用 DCT，我们统一用 FFT）
2. 加入 mask 软门控

---

## 3. 代码目录结构

```
WaveMST/
├── dataset/
│   ├── CAVE_1024_28/            # scene1.npy ~ scene242.npy
│   └── TSA_simu_data/
│       ├── mask.npy             # (256, 256) binary mask
│       └── Truth/               # scene01.npy ~ scene10.npy
│
├── result/
│   ├── model/                   # 自动按 [date_time]_[model_name]/ 组织
│   └── show/                    # test.py 输出，按 [date_time]_test/ 组织
│
├── MST/                         # 参考代码（只读）
├── WaveFormer/                  # 参考代码（只读）
├── vegetation_index/            # 用户已有代码（只读，参考 pipeline 风格）
│
├── dataset.py                   # 数据加载、mask、CASSI 仿真
├── mst.py                       # MST baseline 模型（从 MST repo 复制精简）
├── wpo3d.py                     # Model 0 & Model 1 (3D WPO Pure / KG)
├── wpo_smsa.py                  # Model 2 (3D WPO + S-MSA 并联)
├── wpo_mamba.py                 # Model 3 (2D WPO + 1D Mamba)
├── mask_ops.py                  # 三种 mask 机制 class
├── loss.py                      # PSNR, SSIM, SAM, Params/FLOPs 计算
├── train.py                     # 训练入口（模型选择用索引）
├── test.py                      # 测试入口
└── viz.py                       # 可视化（拼接裁剪块）
```

---

## 4. 各文件实现细节

### 4.1 dataset.py

**功能**：数据加载、mask 生成、CASSI 仿真测量值生成、数据增强。

**参照**：`MST/simulation/train_code/utils.py` 的 `LoadTraining`, `LoadTest`, `generate_masks`, `shift`, `shift_back`, `gen_meas_torch`, `shuffle_crop`, `arguement_1/2`。可以同时参照 `vegetation_index/` 的风格。

**关键函数**：

```python
def load_training(data_path, max_scenes=205):
    """加载 CAVE npy 文件。返回 list of ndarray, 每个 (1024, 1024, 28)"""
    # 支持 .npy 和 .mat 两种格式
    # .npy 直接 np.load
    # .mat 用 scipy.io.loadmat, key 是 'img' 或 'img_expand', 除以 65536

def load_test(test_path):
    """加载测试集。返回 tensor [N, 28, 256, 256]"""

def load_mask(mask_path):
    """加载 mask.npy (256, 256)。返回 mask3d [28, 256, 256]（复制28次）"""

def generate_shift_mask(mask3d, step=2):
    """生成 shifted mask [28, 256, 310]，用于 MST 的 MaskGuidedMechanism"""
    # 参照 MST utils.py shift() 函数

def shift_back(inputs, step=2):
    """从 [B, 28, 256, 310] 裁回 [B, 28, 256, 256]"""
    # 直接复用 MST utils.py shift_back()

def gen_meas(gt, mask3d, input_setting='H'):
    """在线生成 CASSI 测量值
    gt: [B, 28, 256, 256]
    mask3d: [B, 28, 256, 256]
    
    CASSI 过程：
    1. masked = mask3d * gt           # 掩模调制
    2. shifted = shift(masked, step=2)  # 色散偏移（每个波段偏移 step*i 像素）
    3. meas = sum(shifted, dim=1)      # 传感器积分 → [B, 256, 310]
    4. H = shift_back(meas/28*2)       # 初始化估计 → [B, 28, 256, 256]
    """

def shuffle_crop(train_data, batch_size, crop_size=256, augment=True):
    """随机裁剪+数据增强，参照 MST utils.py"""
    # 一半直接裁剪+翻转旋转
    # 一半拼接四个 128x128 块（MST 的 arguement_2 策略）
```

### 4.2 mst.py

**功能**：MST baseline 模型。直接从 `MST/simulation/train_code/architecture/MST.py` 复制，精简不必要的注释。

**包含类**：`MaskGuidedMechanism`, `MS_MSA`, `FeedForward`, `MSAB`, `MST`

**不修改任何逻辑**，仅作为 baseline 对照和 Model 2 的组件复用。

### 4.3 wpo3d.py

**功能**：Model 0 (WaveMST_3D) 和 Model 1 (WaveMST_KG) 的模型定义。

**包含类**：

```python
class WPO3D(nn.Module):
    """3D Wave Propagation Operator — 核心物理传播模块
    
    输入: x [B, C, H, W], mask_spatial [B, C, H, W]
    输出: [B, C, H, W]
    
    可学习参数:
        alpha: 阻尼系数 (per-layer)
        vs:    空间波速
        vl:    光谱波速（作用在 C/通道维度）
        t:     传播时间步长
    """

class WPO3D_KG(WPO3D):
    """Klein-Gordon 扩展的 WPO，继承 WPO3D
    额外参数: m0_sq（质量场基础强度）, kg_weight
    forward 中先调父类得零阶解，再加 Born 修正
    """

class WPO3DBlock(nn.Module):
    """单个 WPO Block = LN + WPO3D + Residual + LN + FFN + Residual
    参照 WaveFormer 的 WaveBlock 结构
    """

class WaveMST_3D(nn.Module):
    """完整 U-Net 模型（Model 0）
    参照 MST 的 __init__ 和 forward：
    - embedding → encoder stages → bottleneck → decoder stages → mapping
    - 每个 stage 有 WPO3DBlock × num_blocks[i]
    - encoder 中对 fea 和 mask 同步下采样
    - decoder 中 skip connection + 上采样
    """
    def __init__(self, dim=28, stage=2, num_blocks=[2,2,2], use_kg=False):
        # use_kg=True 时用 WPO3D_KG 替代 WPO3D (Model 1)
    
    def forward(self, x, input_mask):
        # x: [B, 28, 256, 256] — CASSI 初始化估计
        # input_mask: shifted mask [1, 28, 256, 310] 
        #   或 mask3d [1, 28, 256, 256]（WPO 用的是空间 mask）

class WaveMST_KG(WaveMST_3D):
    """Model 1 — 就是 WaveMST_3D(use_kg=True) 的别名"""
```

**FFN 结构**：可用 MST 的 `FeedForward` 类（Conv1x1→GELU→DWConv3x3→GELU→Conv1x1），或 WaveFormer 的 `Mlp` 类。先用Waveformer的`Mlp`类。

**Mask 处理**：
- MST 传入的是 shifted mask `[1, 28, 256, 310]`，WPO 需要的是 spatial mask `[1, 28, 256, 256]`
- 在 `forward` 中，对 shifted mask 调用 `shift_back` 得到 `[1, 28, 256, 256]`
- 或者传入原始 mask3d（未 shift 的）（但是要shift的）

### 4.4 wpo_smsa.py

**功能**：Model 2 (WaveMST_Parallel) — WPO 并联 S-MSA。

**与 wpo3d.py 的区别**：Block 内部多了 S-MSA 分支。

```python
class WPO_SMSA_Block(nn.Module):
    """并联 Block
    内含:
        self.wpo = WPO3D(...)
        self.smsa = MS_MSA(...)  # 从 mst.py 导入
        self.fusion = nn.Linear(dim*2, dim) 或 gate 网络
    
    forward(x, mask_shifted, mask_spatial):
        out_wpo = self.wpo(x, mask_spatial)
        out_smsa = self.smsa(x, mask_shifted)  # MS_MSA 需要 shifted mask
        out = self.fusion(cat(out_wpo, out_smsa))
        return out
    """

class WaveMST_Parallel(nn.Module):
    """与 WaveMST_3D 相同的 U-Net 骨架，但用 WPO_SMSA_Block 替代 WPO3DBlock"""
    # forward 需要同时传 shifted mask 和 spatial mask
```

### 4.5 wpo_mamba.py

**功能**：Model 3 (WaveMST_Mamba) — 2D WPO + 1D Mamba。

```python
class WPO2D(nn.Module):
    """2D Wave Propagation Operator（参照 WaveFormer Wave2D）
    对每个通道独立做 2D FFT 波传播
    与 WPO3D 区别：只在 (H, W) 两个维度做 FFT，C 维度不参与
    """

class SpectralMamba(nn.Module):
    """1D Mamba 沿通道/光谱维度
    输入: [B, C, H, W] → reshape [B*H*W, C, 1] → Mamba1D → reshape back
    依赖: mamba_ssm 包。如果不可用，退化为 1D Conv 或简单 SSM
    """

class WPO_Mamba_Block(nn.Module):
    """串联: 2D WPO（空间）→ 1D Mamba（光谱）→ Residual → FFN → Residual"""

class WaveMST_Mamba(nn.Module):
    """同样的 U-Net 骨架"""
```

### 4.6 mask_ops.py

**功能**：三种 mask 机制的实现，默认使用 Class1。

```python
class MaskGateA(nn.Module):
    """方案 A：初始振幅软门控（默认）
    
    包含:
        self.eps = 0.1  # 软门控下限
        self.velocity_encoder = nn.Sequential(DWConv, GELU, Conv1x1)  # Ψ 编码器
    
    forward(x, mask_spatial):
        gate = self.eps + (1 - self.eps) * mask_spatial
        u0 = x * gate
        v0 = self.velocity_encoder(x) * gate
        return u0, v0
    """

class MaskSourceB(nn.Module):
    """方案 B：Mask 作为源项
    
    包含:
        self.source_net = Conv2d(...)  # 生成 S
    
    forward(x, mask_spatial, decay, sinc_term):
        S = self.source_net(x)
        source_contribution = mask_spatial * S  
        # 频域: FFT(source_contribution) * sinc_term * decay
        return source_fft_term  # 加到主传播结果上
    """

class MaskKleinGordonD(nn.Module):
    """方案 D：Klein-Gordon 质量场 + Born 近似
    
    包含:
        self.m0_sq = nn.Parameter(torch.tensor(0.1))  # clamped to [0, 0.5]
        self.kg_weight = nn.Parameter(torch.tensor(0.1))
    
    forward(u0_output, mask_spatial, sinc_term, decay):
        m_sq = self.m0_sq.clamp(0, 0.5) * (1 - mask_spatial)
        source = -m_sq * u0_output
        correction = IFFT(FFT(source) * sinc_term * decay)
        return u0_output + self.kg_weight * correction
    """
```

**WPO3D 内部调用**：`WPO3D.__init__` 接受参数 `mask_mode='A'`，内部实例化对应的 mask 类。

### 4.7 loss.py

**功能**：损失函数和评估指标。

```python
def torch_psnr(img, ref):
    """参照 MST utils.py torch_psnr，逐通道计算再平均"""
    # img, ref: [28, 256, 256]

def torch_ssim(img, ref):
    """参照 MST ssim_torch.py"""

def torch_sam(img, ref):
    """Spectral Angle Mapper — 光谱角映射
    SAM = arccos(dot(img, ref) / (||img|| * ||ref||))
    逐像素计算光谱角再平均
    """

def count_params(model):
    """返回参数量 (M)"""

def count_flops(model, input_shape=(1, 28, 256, 256)):
    """返回 FLOPs (G)，用 thop 或 fvcore"""
```

**训练损失**：`loss = torch.sqrt(mse(pred, gt))`（与 MST 一致，RMSE）。

### 4.8 train.py

**设计原则**：不用 argparse。所有可调参数直接写在文件顶部的 CONFIG 区域。模型选择用整数索引。

```python
# ============ CONFIG ============
MODEL_INDEX = 0     # 0: 3D-WPO-Pure, 1: 3D-WPO-KG, 2: 3D-WPO-SMSA, 3: 2D-WPO-Mamba
GPU_ID = '0'
BATCH_SIZE = 5
MAX_EPOCH = 300
LEARNING_RATE = 4e-4
SCHEDULER = 'CosineAnnealingLR'  # or 'MultiStepLR'
MILESTONES = [50, 100, 150, 200, 250]
CROP_SIZE = 256
NUM_BANDS = 28
EPOCH_SAMPLE_NUM = 5000
DATA_ROOT = './dataset'
DIM = 28
STAGE = 2
NUM_BLOCKS = [2, 2, 2]
MASK_MODE = 'A'     # 'A', 'B', 'D'
INPUT_SETTING = 'H'  # 'H', 'HM', 'Y'
# ================================

MODELS = {
    0: ('WaveMST_3D',       '3d_wpo_pure'),
    1: ('WaveMST_KG',       '3d_wpo_kg'),
    2: ('WaveMST_Parallel',  '3d_wpo_smsa'),
    3: ('WaveMST_Mamba',     '2d_wpo_mamba'),
}

# --- 模型实例化 ---
def build_model(index):
    if index == 0:
        from wpo3d import WaveMST_3D
        return WaveMST_3D(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, mask_mode=MASK_MODE)
    elif index == 1:
        from wpo3d import WaveMST_3D
        return WaveMST_3D(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, 
                          mask_mode=MASK_MODE, use_kg=True)
    elif index == 2:
        from wpo_smsa import WaveMST_Parallel
        return WaveMST_Parallel(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, 
                                mask_mode=MASK_MODE)
    elif index == 3:
        from wpo_mamba import WaveMST_Mamba
        return WaveMST_Mamba(dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS, 
                             mask_mode=MASK_MODE)

# --- 训练循环 ---
def train_epoch(epoch, model, optimizer, train_set, mask3d_train, input_mask_train):
    model.train()
    epoch_loss = 0
    batch_num = EPOCH_SAMPLE_NUM // BATCH_SIZE
    t0 = time.time()
    for i in range(batch_num):
        gt = shuffle_crop(train_set, BATCH_SIZE, CROP_SIZE).cuda().float()
        input_meas = gen_meas(gt, mask3d_train, INPUT_SETTING)
        
        optimizer.zero_grad()
        pred = model(input_meas, input_mask_train)
        loss = torch.sqrt(F.mse_loss(pred, gt))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    elapsed = time.time() - t0
    avg_loss = epoch_loss / batch_num
    print(f"[Epoch {epoch:03d}] Loss: {avg_loss:.6f}  Time: {elapsed:.1f}s  "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    return avg_loss

# --- 测试 ---
def test_epoch(epoch, model, test_data, mask3d_test, input_mask_test):
    model.eval()
    with torch.no_grad():
        input_meas = gen_meas(test_data.cuda().float(), mask3d_test, INPUT_SETTING)
        pred = model(input_meas, input_mask_test)
    # 计算 PSNR, SSIM per scene
    # 打印结果
    # 返回 pred, psnr_mean, ssim_mean

# --- main ---
def main():
    model = build_model(MODEL_INDEX).cuda()
    # 打印模型信息
    print(f"Model: {MODELS[MODEL_INDEX][0]}, Params: {count_params(model):.2f}M")
    
    # 数据
    train_set = load_training(...)
    test_data = load_test(...)
    mask3d_train, input_mask_train = load_and_prepare_mask(...)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ...
    
    # 输出目录
    save_dir = f"result/model/{time_str}_{MODELS[MODEL_INDEX][1]}/"
    
    best_psnr = 0
    for epoch in range(1, MAX_EPOCH + 1):
        loss = train_epoch(epoch, model, optimizer, ...)
        psnr_mean, ssim_mean = test_epoch(epoch, model, ...)
        scheduler.step()
        
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            if psnr_mean > 28:  # 达到一定水平才保存
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                print(f"  ★ New best: PSNR={psnr_mean:.2f} SSIM={ssim_mean:.4f}")
```

### 4.9 test.py

```python
# CONFIG
MODEL_INDEX = 0
CHECKPOINT = 'result/model/xxx/best.pth'

# 加载模型
model = build_model(MODEL_INDEX).cuda()
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

# 加载测试数据
test_data = load_test(...)  # [10, 28, 256, 256]
mask3d_test, input_mask_test = load_and_prepare_mask(..., batch_size=10)

# 推理
with torch.no_grad():
    input_meas = gen_meas(test_data, mask3d_test, INPUT_SETTING)
    pred = model(input_meas, input_mask_test)

# 评估
for i in range(10):
    psnr = torch_psnr(pred[i], test_data[i])
    ssim = torch_ssim(pred[i], test_data[i])
    sam  = torch_sam(pred[i], test_data[i])
    print(f"Scene {i+1}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, SAM={sam:.4f}")

# 保存结果
save_dir = f"result/show/{time_str}_test/"
# 保存为 .npy
```

### 4.10 viz.py

**功能**：
1. 可视化单个场景的多波段伪彩色图
2. 可视化光谱曲线对比（pred vs gt）
3. 可视化频域响应（展示 WPO 的频率调制效果）

**拼接逻辑**：测试图是 256×256 裁剪的，不需要拼接（TSA_simu_data 的测试集本身就是 256×256）。如果是更大图像的裁剪块，viz.py 应支持按位置拼接回去。

---

## 5. 关键实现注意事项

### 5.1 3D FFT 的维度约定

PyTorch `torch.fft.rfftn(x, dim=(-3,-2,-1))` 对 tensor 的后三个维度做 3D FFT。

输入 `x: [B, C, H, W]`：
- `dim=-3` 对应 C（通道/光谱维度）
- `dim=-2` 对应 H
- `dim=-1` 对应 W（rfft 的半频谱维度）

输出 shape: `[B, C, H, W//2+1]`（因为 rfft 在最后一维只返回一半）

频率网格构建：
```python
freq_c = torch.fft.fftfreq(C, device=x.device)       # shape [C]
freq_h = torch.fft.fftfreq(H, device=x.device)       # shape [H]  
freq_w = torch.fft.rfftfreq(W, device=x.device)      # shape [W//2+1]

# 广播为 3D: [C, H, W//2+1]
fc = freq_c[:, None, None]
fh = freq_h[None, :, None]
fw = freq_w[None, None, :]

omega_sq = (2*pi)**2 * (vs**2 * (fh**2 + fw**2) + vl**2 * fc**2)
```

### 5.2 NaN 防护

```python
eta = omega_sq - (alpha/2)**2
# 分区处理
pos_mask = eta > 1e-8
neg_mask = eta < -1e-8
# 中间区域 (|eta| < 1e-8) 用泰勒展开近似

omega_d = torch.zeros_like(eta)
omega_d[pos_mask] = torch.sqrt(eta[pos_mask])

gamma = torch.zeros_like(eta)
gamma[neg_mask] = torch.sqrt(-eta[neg_mask])
```

### 5.3 Mask 在 U-Net 各层的下采样

MST 中 mask 和特征同步下采样（Conv2d stride=2）。对于 WPO 的 spatial mask：

```python
# Encoder 中
mask_spatial = mask3d[:, :, :, :256]  # 从 shifted mask 截取 256 宽度
# 或直接用原始 mask3d [B, 28, 256, 256]

# 每层下采样
self.mask_downsample = nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False)
# 注意：mask 下采样后值域可能不在 [0,1]，需要 sigmoid 或 clamp
```

### 5.4 WPO 的参数初始化

```python
self.alpha = nn.Parameter(torch.tensor(0.1))     # 小阻尼
self.vs    = nn.Parameter(torch.tensor(1.0))      # 空间波速
self.vl    = nn.Parameter(torch.tensor(0.5))      # 光谱波速（小一些，因为光谱维度小）
self.t     = nn.Parameter(torch.tensor(1.0))      # 传播时间

# 用 softplus 保证正值
vs_eff = F.softplus(self.vs)
vl_eff = F.softplus(self.vl)
alpha_eff = F.softplus(self.alpha)
t_eff = F.softplus(self.t)
```

### 5.5 MST 的 shift/shift_back 机制

CASSI 系统中，每个波段 $\lambda_i$ 的像素列偏移 $i \times \text{step}$ 个像素（step=2）。

- `shift(mask3d, step=2)`: `[B, 28, 256, 256]` → `[B, 28, 256, 310]`（右边补零，每个波段向右移 2i）
- `shift_back(meas, step=2)`: `[B, 256, 310]` → `[B, 28, 256, 256]`（从第 2i 列开始截取 256 列）

**Model 2 需要 shifted mask**（给 MS_MSA），**所有 Model 的 WPO 需要原始 spatial mask**（未 shift 的 `[B, 28, 256, 256]`）。所以 forward 需要同时接收两者，或者在 forward 内部做转换。

### 5.6 数据格式兼容

MST 原始数据是 `.mat` 格式（scipy.io.loadmat），用户已经改为 `.npy`。dataset.py

---

## 6. 训练 Pipeline 总结

```
1. 加载数据
   train_set = load_training('dataset/CAVE_1024_28/', max_scenes=205)
   test_data = load_test('dataset/TSA_simu_data/Truth/')
   mask = load_mask('dataset/TSA_simu_data/mask.npy')

2. 准备 mask
   mask3d = tile(mask, 28)           # [28, 256, 256]
   mask3d_batch = expand(mask3d, B)  # [B, 28, 256, 256]
   shift_mask = shift(mask3d_batch)  # [B, 28, 256, 310] (for MS_MSA)

3. 每个 epoch
   for i in range(epoch_sample_num // batch_size):
       gt = shuffle_crop(train_set, batch_size)    # [B, 28, 256, 256]
       input_meas = gen_meas(gt, mask3d_batch)     # [B, 28, 256, 256]
       
       pred = model(input_meas, shift_mask)        # forward
       loss = sqrt(MSE(pred, gt))                  # RMSE loss
       loss.backward()
       optimizer.step()

4. 每个 epoch 末尾
   test：推理 10 个测试场景，计算 PSNR/SSIM
   if best → save checkpoint
```

---

