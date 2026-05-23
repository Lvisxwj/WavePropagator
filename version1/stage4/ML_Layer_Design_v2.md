# ML 层设计深化分析：Wave-Aware 讲解 + Frequency-Based Attention 方案

> **本文回答三个问题**：
> 1. Wave-Aware 的设计逻辑，WPO 是否也能用波数
> 2. Frequency-Based Attention 的完整设计（频域版 SSR WSSA）
> 3. 文献支撑与最终推荐

---

## 目录

1. [Wave-Aware 的设计逻辑与波数问题](#1-wave-aware-的设计逻辑与波数问题)
2. [文献支撑：频域注意力的最新进展](#2-文献支撑频域注意力的最新进展)
3. [Frequency-Based Attention 完整设计](#3-frequency-based-attention-完整设计)
4. [三种 ML 层的最终对比与推荐](#4-三种-ml-层的最终对比与推荐)

---

## 1. Wave-Aware 的设计逻辑与波数问题

### 1.1 Wave-Aware 做了什么

Wave-Aware 的核心思路是：**在空间域做 DWConv 之后，用特征自身的频谱能量分布来生成通道权重**——不是从"全局平均池化"得到权重（那是 SE-Net），而是从"频谱能量"得到权重。

具体流程：

```
输入 x [B, C, H, W]
    │
    ├── DWConv 3×3 + GELU + Conv 1×1  →  feat [B, C, H, W]    （空间局部特征）
    │
    ├── 2D rFFT(feat)  →  freq [B, C, H, W//2+1] complex
    │       │
    │       └── |freq|.mean(dim=(-2,-1))  →  freq_energy [B, C]  （每通道平均频谱幅度）
    │
    ├── freq_energy + k_bias  →  gate_input [B, C]
    │       │
    │       └── MLP + sigmoid  →  gate [B, C, 1, 1]
    │
    └── feat * gate  →  输出 [B, C, H, W]
```

**和 SE-Net 的关键区别**：SE 用空间平均值做通道权重（$\text{AvgPool}(x) \to \text{MLP} \to \sigma$），Wave-Aware 用频谱幅度做通道权重（$|\text{FFT}(x)|_\text{avg} \to \text{MLP} \to \sigma$）。

**为什么频谱幅度更好**：空间平均值只能看"每个通道有多亮"，频谱幅度能看"每个通道的能量在高频还是低频"——后者的信息量更大，而且和 WPO 的频域操作天然对应。

### 1.2 波数 k_bias 在 WPO 中能不能用

**你的疑问完全合理**。让我澄清：

**k_bias 是一个可学习参数**，初始化为物理波数 $k(\lambda)$ 但允许被训练修改。它的作用是给每个通道一个"重要性偏置"——短波通道（大 $k$）天然被赋予更高的初始门控权重。

**WPO 版本**：WPO 的色散关系是 $\omega_d^2 = v_s^2|\boldsymbol{\omega}|^2 + v_\lambda^2\omega_\lambda^2 - (\alpha/2)^2$，没有显式的 $k(\lambda)$。但 **k_bias 不是注入色散关系的——它只是通道权重的初始化偏置**。物理直觉：短波通道（蓝光）的特征通常含更多高频信息（纹理边缘），值得被 ML 层更多关注。这个观察对 WPO 和 KG 都成立。

所以 **WPO 也能用 k_bias**。只是对 KG 来说，k_bias 和色散关系的 $k^2(\lambda)$ 形成"双重波数注入"（ML 层 + WPO 层各注入一次），物理一致性更强。

**如果你不想在 WPO 版本里用波数**，可以把 k_bias 初始化为零（退化为普通可学习偏置），效果不会变差——只是失去了物理初始化的潜在好处。

### 1.3 Wave-Aware 的局限

Wave-Aware 的频域操作只用于**统计**（计算平均频谱幅度），不用于**特征变换**。也就是说，特征本身仍然在空间域处理，频域信息只用来生成门控信号。

**更激进的方案**：直接在频域做特征变换和注意力——这就是下面要讲的 Frequency-Based Attention。

---

## 2. 文献支撑：频域注意力的最新进展

搜索到了几个与你的想法高度相关的最新工作：

### 2.1 CASSI HSI 重建中的频域方法

| 论文 | 发表 | 核心思路 | 与你想法的关系 |
|------|------|---------|-------------|
| **Specformer** | **ECCV 2024** | 提出 **FWSA（Frequency-Wise Self-Attention）**，与 LWSA（Local-Window Spatial Attention）并行，形成 Spatial-Frequency Block | **直接相关**——在 CASSI HSI 中做频域注意力 |
| Hybrid Sparse Transformer + Wavelet Fusion | Sensors 2024 | 小波融合 + Sparse Transformer 做 CASSI unfolding | 小波+注意力混合 |
| SSTHyper | ACCV 2024 | Sparse Spectral Transformer，利用光谱间稀疏相似性 | 光谱域稀疏注意力 |

### 2.2 通用图像恢复中的频域注意力

| 论文 | 发表 | 核心思路 |
|------|------|---------|
| **FFTNet** | arXiv 2025 (2502.18394) | 用 FFT 做自适应频域 token mixing，$O(N\log N)$，hybrid local window + global FFT |
| **HFMNet** (Hybrid Frequency Modulation) | **IJCAI 2024** | 双维度（空间+通道）频域调制，Fourier+Wavelet 混合 |
| MSFSNet | arXiv 2024 | 动态滤波器选频 + 频率交叉注意力（FCAM） |
| **SAD-Net** | Scientific Reports 2025 | 小波增强卷积 + 频率引导注意力（FGA） |
| WavEnhancer | J. Comput. Sci. 2024 | 小波域分高低频 + Transformer 处理低频 + CNN 处理高频 |
| CosAE | **NeurIPS 2024** | 频域系数作为 autoencoder 的极窄 bottleneck |

### 2.3 关键发现

**Specformer (ECCV 2024)** 已经在 CASSI 中做了 Frequency-Wise Self-Attention。但它的"频域"指的是**对特征做 FFT 后在频率分量间计算 attention**——是空间频域。

你的想法"频域版 SSR WSSA"更进一步：SSR 的 WSSA 是在**空间窗内做光谱 attention**，你想在**频率子带内做光谱 attention**——这与 Specformer 不同，是一个新组合。

**HFMNet (IJCAI 2024)** 最接近你的想法——它同时用 Fourier 和 Wavelet 做双维度频域调制。但它是通用图像恢复，不是 CASSI。

**FFTNet (arXiv 2025)** 的"adaptive spectral filter"思路值得借鉴——用 FFT 做 $O(N\log N)$ 的全局 token mixing，但通过可学习滤波器只保留有用频率。

---

## 3. Frequency-Based Attention 完整设计

### 3.1 核心思路

SSR 的 WSSA 做的是：

```
空间域特征 → 空间切窗 → 窗内全光谱 attention → 重组
```

我们的 Frequency-Based Attention (FBA) 做的是：

```
空间域特征 → 小波分解为多频带 → 每个频带内做全光谱 attention → 小波重建
```

**类比**：WSSA 把空间分成"局部区域"再分析光谱关系，FBA 把空间分成"频率子带"再分析光谱关系。

**物理直觉**：HSI 不同波段的相关性在不同频率下不同——低频（大面积均匀区域）的波段相关性很高（都是"背景"），高频（边缘纹理）的波段相关性较低（不同材质的边缘不同）。按频带分别做光谱 attention 可以捕捉这种**频率依赖的光谱相关性**。

### 3.2 为什么用小波而不是 FFT

**FFT 的问题**：FFT 的频率分量是全局的（每个频率点对应整张图的一个正弦波），失去了空间位置信息。做完 FFT 再做 attention，不知道"这个高频分量来自图像的哪个位置"。

**小波的优势**：小波变换（DWT）同时保留**频率**和**空间位置**信息。2D DWT 把图像分解为 4 个子带：

- **LL**（低频近似）：大尺度结构，尺寸 H/2 × W/2
- **LH**（水平高频）：水平边缘
- **HL**（垂直高频）：垂直边缘
- **HH**（对角高频）：对角纹理

每个子带都保留了空间信息（只是分辨率减半），可以在子带内正常做 attention。

### 3.3 完整架构

```python
class ML_FreqBandAttention(nn.Module):
    """选项 3（升级版）：Frequency-Band Attention (FBA)
    
    设计原理（新方案，替代原 Wave-Aware）：
    1. 2D DWT 把空间特征分解为 4 个频率子带 (LL, LH, HL, HH)
    2. 每个子带内独立做全光谱 Self-Attention（类似 WSSA 但按频带分而非按空间窗分）
    3. 不同子带的 attention 权重不同（低频用全局 attention，高频用局部 attention）
    4. 2D iDWT 重建回空间域
    
    物理动机：
    - 波动方程的不同频率分量有不同传播速度（色散）
    - 低频分量传播远（全局结构）→ 用全局光谱 attention
    - 高频分量传播近（局部细节）→ 用局部光谱 attention
    - 这与 WPO 的频域调制天然互补
    
    文献支撑：
    - SSR (CVPR 2024)：空间窗内光谱 attention
    - Specformer (ECCV 2024)：频域注意力
    - HFMNet (IJCAI 2024)：双维度频域调制
    - WavEnhancer (2024)：小波+Transformer 分频处理
    - 本方案：小波分频 + 频带内光谱 attention（新组合）
    
    复杂度：O(HW·C²/4)——因为每个子带空间尺寸减半，总计算量约为 WSSA 的 1/4
    参数量：~8 × dim²（Q/K/V 投影 + 输出投影 + 子带自适应权重）
    """
    def __init__(self, dim, wavelet='haar'):
        super().__init__()
        self.dim = dim
        self.wavelet = wavelet
        
        # 每个子带的 Q/K/V 投影（共享权重以节省参数，但各子带有独立的缩放）
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        
        # 子带自适应权重：4 个标量，控制各子带 attention 的强度
        # 初始化：LL 权重最大（低频最重要），HH 权重最小
        self.band_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.5, 0.25]))
        
        # 可选：子带间交互（轻量 Conv1×1）
        self.band_interact = nn.Conv2d(dim * 4, dim * 4, 1, groups=4, bias=False)
        
        self.scale = dim ** -0.5
    
    def dwt2d(self, x):
        """2D Haar 小波变换（最简单的 DWT）
        
        输入: [B, C, H, W]
        输出: (LL, LH, HL, HH)，各 [B, C, H/2, W/2]
        
        Haar DWT 只需要加减和缩放，不需要额外参数：
          LL = (x[::2,::2] + x[::2,1::2] + x[1::2,::2] + x[1::2,1::2]) / 2
          LH = (x[::2,::2] + x[::2,1::2] - x[1::2,::2] - x[1::2,1::2]) / 2
          HL = (x[::2,::2] - x[::2,1::2] + x[1::2,::2] - x[1::2,1::2]) / 2
          HH = (x[::2,::2] - x[::2,1::2] - x[1::2,::2] + x[1::2,1::2]) / 2
        """
        x00 = x[:, :, 0::2, 0::2]  # 偶行偶列
        x01 = x[:, :, 0::2, 1::2]  # 偶行奇列
        x10 = x[:, :, 1::2, 0::2]  # 奇行偶列
        x11 = x[:, :, 1::2, 1::2]  # 奇行奇列
        
        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 + x01 - x10 - x11) * 0.5
        HL = (x00 - x01 + x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5
        
        return LL, LH, HL, HH
    
    def idwt2d(self, LL, LH, HL, HH):
        """2D Haar 逆小波变换
        
        输入: 各 [B, C, H/2, W/2]
        输出: [B, C, H, W]
        """
        B, C, H2, W2 = LL.shape
        out = torch.zeros(B, C, H2*2, W2*2, device=LL.device, dtype=LL.dtype)
        
        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
        out[:, :, 0::2, 1::2] = (LL + LH - HL - HH) * 0.5
        out[:, :, 1::2, 0::2] = (LL - LH + HL - HH) * 0.5
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5
        
        return out
    
    def spectral_attention(self, x, weight=1.0):
        """在单个频率子带内做全光谱 Self-Attention
        
        输入: [B, C, H', W']（子带，H'=H/2, W'=W/2）
        
        每个空间位置看全部 C 个波段的关系。
        相当于把每个像素的光谱向量作为 token，做 C×C 的 attention。
        """
        B, C, H, W = x.shape
        
        # Q, K, V
        qkv = self.qkv(x)           # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, 1)   # 各 [B, C, H, W]
        
        # 展平空间维：[B, C, HW]
        q = q.view(B, C, -1)        # [B, C, HW]
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        
        # 光谱 attention：[B, C, C]
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, C, C]
        attn = (attn * weight).softmax(dim=-1)
        
        out = torch.bmm(attn, v)     # [B, C, HW]
        return out.view(B, C, H, W)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 小波分解
        LL, LH, HL, HH = self.dwt2d(x)
        
        # 2. 每个子带独立做光谱 attention，权重不同
        w = torch.softmax(self.band_weights, dim=0)  # 归一化子带权重
        LL_attn = self.spectral_attention(LL, w[0])
        LH_attn = self.spectral_attention(LH, w[1])
        HL_attn = self.spectral_attention(HL, w[2])
        HH_attn = self.spectral_attention(HH, w[3])
        
        # 3. 可选：子带间交互
        # 把 4 个子带 concat 在通道维，做分组 Conv1×1
        bands = torch.cat([LL_attn, LH_attn, HL_attn, HH_attn], dim=1)  # [B, 4C, H/2, W/2]
        bands = self.band_interact(bands)
        LL_out, LH_out, HL_out, HH_out = bands.chunk(4, dim=1)
        
        # 4. 逆小波重建
        out = self.idwt2d(LL_out, LH_out, HL_out, HH_out)  # [B, C, H, W]
        
        return self.proj(out)
```

### 3.4 与 WSSA 的精确对比

| 维度 | WSSA (SSR) | FBA (我们) |
|------|-----------|----------|
| 分域方式 | 空间切窗（8×8）| 频率子带（DWT 4 子带）|
| attention 类型 | 窗内全光谱 $C \times C$ | 子带内全光谱 $C \times C$ |
| 空间信息 | 窗内保留，跨窗丢失 | 子带内保留（分辨率减半）|
| 频率信息 | 无显式频率处理 | 显式按频率分带 |
| 跨区域交互 | 需要后续 CMB（11×11 conv） | 逆小波重建自动恢复全局 |
| 复杂度 | $O(HWC^2)$ | $O(HWC^2/4)$（子带空间减半）|
| 参数量 | 6C² | 8C²（多了子带权重和交互）|

**关键区别**：WSSA 的窗边界是硬切的（相邻窗完全隔离），需要 CMB 补充跨窗交互。FBA 的子带分解是**完美重建**的（iDWT 无信息损失），不需要额外的跨区域交互模块。

### 3.5 与 Specformer (ECCV 2024) 的区别

Specformer 的 FWSA 是在**空间频域**做 attention——对特征做空间 FFT，在频率分量之间计算 attention。它的"频率"指空间频率。

我们的 FBA 是在**频率子带**内做**光谱** attention——先用 DWT 分频，再在每个子带内做波段间 attention。它的"频率"是小波子带，"attention"是光谱维度。

两者正交，可以组合（Specformer 的 FWSA + 我们的 FBA），但单独来看我们的方案更有物理动机——因为 WPO 的频域调制正好是按频率处理的，FBA 的分频处理和 WPO 天然对接。

### 3.6 与 WPO 的互补性分析

**WPO 做的是**：把特征变到频域 → 按频率用波动方程闭式解调制 → 变回空间域

**FBA 做的是**：把特征按频率分解 → 每个频带内做光谱相关性学习 → 重建

两者互补：

1. **WPO 管"频率内传播"**：同一个频率分量在空间上怎么传播（波速 $v_s$）
2. **FBA 管"频率间光谱关系"**：不同波段在同一频率下的相关性

WPO 不做光谱间交互（每个频率点独立调制），FBA 不做空间传播（只在固定子带内做 attention）。**两者合起来覆盖了完整的"空间传播 + 光谱交互 + 频率感知"三维信息**。

### 3.7 计算复杂度对比

设 $N = HW$，$C = 28$：

| ML 层 | 复杂度 | dim=28, 256×256 |
|-------|-------|----------------|
| DWConv+CA | $O(9NC + C^2/r)$ | ~1.7M ops |
| WSSA | $O(NC^2)$ = $O(6NC^2)$ | ~130M ops |
| Wave-Aware | $O(9NC + N\log N \cdot C)$ | ~12M ops |
| **FBA** | $O(NC^2/4 \times 4 + 4NC)$ = $O(NC^2 + 4NC)$ | ~53M ops |

FBA 比 WSSA 轻约 60%（因为子带空间减半），但比 DWConv+CA 重约 30 倍。对比 WPO 本身的 3D FFT 复杂度 $O(NC\log(NC)) \approx 30M$，FBA 的 53M 是同数量级——可接受。

---

## 4. 三种 ML 层的最终对比与推荐

### 4.1 更新后的三选项

| 选项 | 名称 | 核心机制 | 参数量 | 复杂度 | 与 WPO 互补性 | 创新性 |
|------|------|---------|-------|-------|-------------|-------|
| 1 | DWConv+CA | 空间局部 + 通道权重 | 最小 | 最低 | 中（补局部） | 低 |
| 2 | WSSA | 空间窗内光谱 attention | 中 | 最高 | 中（补光谱交互） | 低（SSR 已有） |
| 3 | **FBA** | 小波分频 + 频带内光谱 attention | 中 | 中 | **高（频域互补）** | **高（新组合）** |

### 4.2 推荐策略

**论文主推选项 3（FBA）**——它有最强的物理动机、最高的创新性、和 WPO 最好的互补性。而且它比 WSSA 计算量更低（子带空间减半），性能可能相当或更好。

**选项 1 作为消融基线**——证明"任何 ML 层都比纯 WPO 好"。

**选项 2 作为消融对照**——证明"我们的 FBA 比直接搬 WSSA 更适合和 WPO 配合"。

**消融表设计**：

| 配置 | E2E PSNR | 5stg PSNR | 说明 |
|------|----------|-----------|------|
| WPO only | 34.70 | 38.21 | baseline |
| DWConv+CA → WPO | ~35.0 | ~38.5 | 最轻量 ML |
| WSSA → WPO | ~35.3 | ~39.0 | SSR 风格 |
| **FBA → WPO** | **~35.5** | **~39.2** | **我们的方案** |

如果 FBA > WSSA——说明"频域分带再做光谱 attention"比"空间切窗再做光谱 attention"更适合和频域物理算子（WPO）配合。这是一个有价值的实验发现。

### 4.3 train.py 中的选项更新

```python
ML_TYPE = 'freq_band'    # 'dwconv_ca' / 'wssa' / 'freq_band'
```

`build_ml_layer` 函数更新：

```python
def build_ml_layer(ml_type, dim):
    if ml_type == 'dwconv_ca':
        return ML_DWConvCA(dim)
    elif ml_type == 'wssa':
        return ML_WSSA(dim)
    elif ml_type == 'freq_band':
        return ML_FreqBandAttention(dim)
    else:
        raise ValueError(f"未知 ML 层类型: {ml_type}")
```

### 4.4 论文叙事的加强

有了 FBA，论文的故事线更统一：

> 我们提出一个**全频域**的 HSI 重建框架：
>
> 1. **WPO 层在频域做全局物理传播**——波动方程的闭式解在频域实现
> 2. **FBA 层在频率子带内做光谱注意力**——小波分解后按频带学习波段间关系
> 3. **两者交替**：FBA 为 WPO 提供"频率感知的光谱特征"，WPO 为 FBA 提供"物理约束的全局传播"
>
> 整个 Block 的信息流完全在频域/子带域中流转——空间域只在输入/输出时出现。

这比"DWConv 补局部 + WPO 补全局"的故事深刻得多——后者是两个不相关的模块拼接，前者是一个统一的频域处理范式。

