# Stage 4：ML-WPO 混合架构 — Claude Code 技术交接

> **目的**：在 stage2/ 代码基础上，新建 stage4/ 目录，实现 ML 层与 WPO 层交替的混合架构。用户通过 train.py 顶部配置选择 ML 层类型和 U-Net 骨架类型。
>
> **前置条件**：stage2/ 的 3D-WPO 5-stage unfolding 已跑通（232epoch, 38.21 dB）。本次不修改 WPO 内部，只在 Block 层级加入 ML 层。
>
> **参考仓库**：`./SSR/` 已 clone 到本地。

---

## 目录

1. [文件操作清单](#1-文件操作清单)
2. [新增文件：ml_layers.py](#2-新增文件ml_layerspy)
3. [修改文件：wpo3d.py](#3-修改文件wpo3dpy)
4. [修改文件：wpo3d_unfold.py](#4-修改文件wpo3d_unfoldpy)
5. [修改文件：train.py](#5-修改文件trainpy)
6. [关键陷阱与调试](#6-关键陷阱与调试)
7. [开发与验证顺序](#7-开发与验证顺序)

---

## 1. 文件操作清单

### 1.1 第一步：从 stage2/ 复制需要的文件到 stage4/

```bash
mkdir -p stage4
cp stage2/wpo3d.py        stage4/
cp stage2/wpo3d_unfold.py stage4/
cp stage2/unfolding_ops.py stage4/
cp stage2/mask_ops.py     stage4/
cp stage2/dataset.py      stage4/
cp stage2/loss.py         stage4/
cp stage2/physics.py      stage4/
cp stage2/train.py        stage4/
cp stage2/test.py         stage4/
cp stage2/enhancement_ops.py stage4/   # 色散介质（保留）
```

不复制的：`helm_pure.py`, `helmholtz_ops.py`, `wpo3d_phys.py`, `wpo3d_helm.py`, `wpo_smsa.py`, `wpo_mamba.py`, `mst.py`, `viz.py`——stage4 不需要这些。

### 1.2 第二步：新增文件

```
stage4/ml_layers.py       ← 新增：三种 ML 层实现
```

### 1.3 第三步：修改文件

```
stage4/wpo3d.py           ← 修改：新增 ML_WPO_Block，新增 WaveMST_ML 模型类
stage4/wpo3d_unfold.py    ← 修改：新增 WaveMST_ML_Unfold
stage4/train.py           ← 修改：重写 CONFIG 和 MODELS，删除不必要选项，print 组合
```

---

## 2. 新增文件：ml_layers.py

### 2.1 文件结构概览

```python
# ml_layers.py — 三种 ML 层 + 统一接口
#
# 选项 1: ML_DWConvCA     — DWConv + Channel Attention（轻量）
# 选项 2: ML_WSSA         — Window-based Spectral Self-Attention（参照 SSR）
# 选项 3: ML_WaveAware    — 频域感知卷积（我们自己设计，贴合波动方程）
#
# 统一接口: build_ml_layer(ml_type, dim) → nn.Module
#   forward(x) → x，输入输出 shape 都是 [B, C, H, W]
```

### 2.2 选项 1：ML_DWConvCA（DWConv + Channel Attention）

最轻量的方案。空间用 DWConv 局部感知，通道用 SE-like attention 做波段重要性加权。

```python
class ML_DWConvCA(nn.Module):
    """选项 1：DWConv + Channel Attention
    
    DWConv3×3 提取空间局部特征 → GELU → Conv1×1 通道混合
    → SE-style Channel Attention（squeeze-excitation）
    
    参数量：~3 × dim² + 2 × dim²/r ≈ 3.5 × dim²
    dim=28 时约 2.7K 参数
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),   # DWConv
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),                      # 通道混合
        )
        # SE channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        feat = self.spatial(x)
        attn = self.se(feat)
        return feat * attn
```

### 2.3 选项 2：ML_WSSA（Window-based Spectral Self-Attention）

参照 SSR（`SSR/Model.py` 的 WSSA 实现）。在空间切窗后做全光谱 attention。

**关键参考**：SSR/Model.py 的 `class SSA(nn.Module)` 和 `SSAB`。

```python
class ML_WSSA(nn.Module):
    """选项 2：Window-based Spectra-wise Self-Attention
    
    参照 SSR (CVPR 2024) 的 WSSA 设计。
    空间切成 M×M 窗，每个窗内做完整 C 维光谱注意力。
    
    输入 [B, C, H, W] → 切窗 → 每窗内 spectral attention → 重组
    
    参考: SSR/Model.py 中的 SSA 类
    
    参数量：~6 × dim² ≈ 4.7K（dim=28）
    """
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        # Q, K, V 投影（各 dim → dim）
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.scale = dim ** -0.5   # 注意力缩放因子
    
    def forward(self, x):
        B, C, H, W = x.shape
        M = self.window_size
        
        # 1. 生成 Q, K, V
        qkv = self.qkv(x)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # 各 [B, C, H, W]
        
        # 2. 空间切窗：[B, C, H, W] → [B*nH*nW, C, M, M]
        nH, nW = H // M, W // M
        
        def window_partition(t):
            # [B, C, H, W] → [B, C, nH, M, nW, M] → [B*nH*nW, C, M, M]
            return t.view(B, C, nH, M, nW, M).permute(0, 2, 4, 1, 3, 5).reshape(B*nH*nW, C, M, M)
        
        q_w = window_partition(q)  # [B*nW, C, M, M]
        k_w = window_partition(k)
        v_w = window_partition(v)
        
        # 3. 每窗内做光谱 attention
        # 把空间展平：[B*nW, C, M²] → 视 C 个 token，每个 token 维度 M²
        BnW = q_w.shape[0]
        q_flat = q_w.view(BnW, C, M*M)   # [BnW, C, M²]
        k_flat = k_w.view(BnW, C, M*M)
        v_flat = v_w.view(BnW, C, M*M)
        
        # attention: [BnW, C, C]
        attn = torch.bmm(q_flat, k_flat.transpose(1, 2)) * (M * self.scale)
        attn = attn.softmax(dim=-1)
        
        out = torch.bmm(attn, v_flat)  # [BnW, C, M²]
        out = out.view(BnW, C, M, M)
        
        # 4. 窗重组：[B*nH*nW, C, M, M] → [B, C, H, W]
        out = out.view(B, nH, nW, C, M, M).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # 5. 输出投影
        return self.proj(out)
```

**注意**：SSR 原版的 WSSA 比这个更精细（有 mask 嵌入、spectral weight 等），需要从G:\MachineLearning\CASSI\stage2\SSR中了解到具体的实现。

### 2.4 选项 3：ML_WaveAware（频域感知卷积，自主设计）

**设计动机**：WPO 在频域做全局传播，ML 层在空间域做局部特征提取——两者的信息域完全分离。设计一个"桥梁"层，在空间域工作但带有频域感知性质，让 ML 输出更适合作为 WPO 的输入。

**核心想法**：DWConv 提取空间局部特征后，用**频率调制门控**做通道加权——门控信号不是从数据学的（那是 SE），而是从特征的**频谱能量分布**计算的。

物理直觉：不同空间频率的特征对 WPO 的响应不同（低频传播远、高频衰减快）。让 ML 层"知道"自己提取的特征在频域的位置，这样它可以针对性地增强那些 WPO 能有效利用的频率分量。

```python
class ML_WaveAware(nn.Module):
    """选项 3：频域感知卷积（Wave-Aware Conv）
    
    设计原理：
    1. DWConv 空间局部特征提取（和选项1相同）
    2. 对特征做快速 2D FFT，计算每个通道的频谱能量分布
    3. 用频谱能量分布生成通道权重（替代 SE 的全局平均池化）
    4. 加上可学习的物理波数偏置 k(λ)（和 KG 方程共享思想）
    
    物理动机：
    - WPO 的频域调制在不同频率有不同增益（共振 vs 过阻尼）
    - ML 层提前"适配"这些频率特性，让 WPO 更有效
    - 波数偏置 k(λ) 让不同波段被不同程度地激活
    
    参数量：~3 × dim² + dim + dim ≈ 2.4K（dim=28）
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 空间局部特征
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        
        # 频谱能量 → 通道权重
        self.freq_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Sigmoid(),
        )
        
        # 物理波数偏置（可选，和 KG 方程思想一致）
        # k_bias[c] 越大 → 该通道被更强激活（短波蓝光 k 大，对应高频特征活跃）
        self.k_bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 空间局部特征
        feat = self.spatial(x)     # [B, C, H, W]
        
        # 2. 计算每通道的频谱能量
        # 只需要统计量，不需要逐频率细节，所以用均值即可
        feat_fft = torch.fft.rfft2(feat, dim=(-2, -1))  # [B, C, H, W//2+1] complex
        freq_energy = feat_fft.abs().mean(dim=(-2, -1))  # [B, C] 每通道平均频谱幅度
        
        # 3. 频谱能量 + 物理偏置 → 通道权重
        # k_bias 可以初始化为归一化物理波数（PhysicsParams 提供）
        gate_input = freq_energy + self.k_bias.view(1, -1)  # [B, C]
        gate = self.freq_proj(gate_input)                     # [B, C]
        gate = gate.view(B, C, 1, 1)                          # [B, C, 1, 1]
        
        return feat * gate
```

**选项 3 的 k_bias 初始化**：

```python
# 在模型 __init__ 中，如果使用 ML_WaveAware：
from physics import PhysicsParams
k_phys = PhysicsParams.get_k_phys_normalized(dim)
for module in model.modules():
    if isinstance(module, ML_WaveAware):
        module.k_bias.data = k_phys  # 用归一化物理波数初始化
```

如果 `physics.py` 中没有 `get_k_phys_normalized`，用简单版本：
```python
wavelengths = torch.linspace(450, 680, dim)
k_phys = 450.0 / wavelengths   # 归一化到 [0.66, 1.0]
```

### 2.5 统一构建接口

```python
def build_ml_layer(ml_type, dim):
    """根据类型名构建 ML 层
    
    Args:
        ml_type: 'dwconv_ca' | 'wssa' | 'wave_aware'
        dim: 通道数
    Returns:
        nn.Module，forward(x) → x，[B, C, H, W] → [B, C, H, W]
    """
    if ml_type == 'dwconv_ca':
        return ML_DWConvCA(dim)
    elif ml_type == 'wssa':
        return ML_WSSA(dim)
    elif ml_type == 'wave_aware':
        return ML_WaveAware(dim)
    else:
        raise ValueError(f"未知 ML 层类型: {ml_type}，可选: dwconv_ca / wssa / wave_aware")
```

---

## 3. 修改文件：wpo3d.py

### 3.1 新增 ML_WPO_Block

在 `WPO3DBlock` 类**之后**，新增：

```python
class ML_WPO_Block(nn.Module):
    """ML-WPO 交替 Block：ML 层 → WPO 层 → FFN
    
    ML 层提取局部空间-光谱特征，WPO 层做物理全局传播，FFN 做通道混合。
    三者各有残差连接。
    
    Args:
        dim:          通道数
        ml_type:      ML 层类型 ('dwconv_ca' / 'wssa' / 'wave_aware')
        mask_mode:    WPO 的 mask 模式
        use_dispersive: WPO 内部是否启用色散修正
    """
    def __init__(self, dim, ml_type='dwconv_ca', mask_mode='A', use_dispersive=False):
        super().__init__()
        from ml_layers import build_ml_layer
        
        # ML 层
        self.norm_ml = LayerNorm2d(dim)
        self.ml = build_ml_layer(ml_type, dim)
        
        # WPO 层（复用已有的 WPO3D）
        self.norm_wpo = LayerNorm2d(dim)
        self.wpo = WPO3D(dim, mask_mode=mask_mode, use_dispersive=use_dispersive)
        
        # FFN
        self.norm_ffn = LayerNorm2d(dim)
        self.ffn = FFN(dim)
    
    def forward(self, x, mask_spatial):
        # Step 1: ML 层（局部特征学习）
        x = x + self.ml(self.norm_ml(x))
        
        # Step 2: WPO 层（物理全局传播，需要 mask）
        x = x + self.wpo(self.norm_wpo(x), mask_spatial)
        
        # Step 3: FFN（通道/光谱混合）
        x = x + self.ffn(self.norm_ffn(x))
        
        return x
```

### 3.2 新增模型类

需要三种 U-Net 配置。用一个统一的模型类 `WaveMST_ML` 通过参数切换：

```python
class WaveMST_ML(nn.Module):
    """ML-WPO 混合模型
    
    U-Net 骨架，Block 类型由 unet_mode 决定：
      - 'symmetric':       所有层用 ML_WPO_Block（对称，最简单）
      - 'asymmetric':      encoder 用 ML_WPO_Block，decoder 用 WPO3DBlock
      - 'alternating':     encoder/decoder 交替 ML_WPO_Block 和 WPO3DBlock
    
    Args:
        dim:        基础通道数
        stage:      U-Net 编码层数
        num_blocks: 每层 block 数
        ml_type:    ML 层类型
        unet_mode:  U-Net 骨架模式
        mask_mode:  WPO 的 mask 模式（'A' 或 'D' for KG）
        use_kg:     True 时强制 mask_mode='D'
        use_dispersive_block: 色散修正
    """
    def __init__(self, dim=28, stage=2, num_blocks=[2,2,2],
                 ml_type='dwconv_ca', unet_mode='symmetric',
                 mask_mode='A', use_kg=False, use_dispersive_block=False):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.unet_mode = unet_mode
        
        if use_kg:
            mask_mode = 'D'
        self.mask_mode = mask_mode
        
        # Embedding
        self.embedding = nn.Conv2d(28, dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        
        def _make_block(d, is_encoder_side=True):
            """根据 unet_mode 决定用哪种 Block"""
            if unet_mode == 'symmetric':
                # 所有层都用 ML_WPO_Block
                return ML_WPO_Block(d, ml_type, mask_mode, use_dispersive_block)
            elif unet_mode == 'asymmetric':
                # encoder 用 ML_WPO，decoder 用纯 WPO
                if is_encoder_side:
                    return ML_WPO_Block(d, ml_type, mask_mode, use_dispersive_block)
                else:
                    return WPO3DBlock(d, mask_mode, use_dispersive_block)
            elif unet_mode == 'alternating':
                # 所有层用 ML_WPO_Block（与 symmetric 相同，
                # 但 Block 内部 ML 和 WPO 的顺序交替——
                # 这里简化为和 symmetric 一样，因为 Block 内部已经是 ML→WPO→FFN）
                return ML_WPO_Block(d, ml_type, mask_mode, use_dispersive_block)
            else:
                raise ValueError(f"未知 unet_mode: {unet_mode}")
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_stage = dim
        for i in range(stage):
            blocks = nn.ModuleList([
                _make_block(dim_stage, is_encoder_side=True)
                for _ in range(num_blocks[i])
            ])
            fea_down  = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            mask_down = nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            self.encoder_layers.append(nn.ModuleList([blocks, fea_down, mask_down]))
            dim_stage *= 2
        
        # Bottleneck（始终用 ML_WPO_Block，这是最深层，需要最强的特征学习）
        self.bottleneck = nn.ModuleList([
            ML_WPO_Block(dim_stage, ml_type, mask_mode, use_dispersive_block)
            for _ in range(num_blocks[-1])
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(stage):
            fea_up = nn.ConvTranspose2d(dim_stage, dim_stage // 2, 2, 2, 0)
            fusion = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)
            blocks = nn.ModuleList([
                _make_block(dim_stage // 2, is_encoder_side=False)
                for _ in range(num_blocks[stage - 1 - i])
            ])
            self.decoder_layers.append(nn.ModuleList([fea_up, fusion, blocks]))
            dim_stage //= 2
        
        # 输出映射
        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)
    
    def forward(self, x, input_mask):
        H = x.shape[2]
        if input_mask.shape[-1] > H:
            mask_spatial = input_mask[:, :, :, :H]
        else:
            mask_spatial = input_mask
        
        fea = self.lrelu(self.embedding(x))
        
        # Encoder
        fea_encoder = []
        masks_enc = []
        for blocks, fea_down, mask_down in self.encoder_layers:
            for blk in blocks:
                fea = blk(fea, mask_spatial)
            fea_encoder.append(fea)
            masks_enc.append(mask_spatial)
            fea = fea_down(fea)
            mask_spatial = torch.sigmoid(mask_down(mask_spatial))
        
        # Bottleneck
        for blk in self.bottleneck:
            fea = blk(fea, mask_spatial)
        
        # Decoder
        for i, (fea_up, fusion, blocks) in enumerate(self.decoder_layers):
            fea = fea_up(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask_spatial = masks_enc[self.stage - 1 - i]
            for blk in blocks:
                fea = blk(fea, mask_spatial)
        
        return self.mapping(fea) + x
```

**注意**：`WaveMST_ML` 和原 `WaveMST_3D` 的 forward 签名完全一致（`x, input_mask`），所以可以无缝替换到 unfolding 包装中。

---

## 4. 修改文件：wpo3d_unfold.py

### 4.1 新增 WaveMST_ML_Unfold

在文件末尾新增，复用现有 `WaveMST_3D_Unfold` 的结构：

```python
class WaveMST_ML_Unfold(WaveMST_3D_Unfold):
    """ML-WPO 混合模型的 unfolding 版本
    
    和 WaveMST_3D_Unfold 完全相同的 unfolding 循环，
    只是 prior network 从 WaveMST_3D 换成 WaveMST_ML。
    """
    def __init__(self, dim=28, stage=2, num_blocks=None,
                 num_stages=5, share_weights=False,
                 ml_type='dwconv_ca', unet_mode='symmetric',
                 mask_mode='A', use_kg=False,
                 size=256, len_shift=2,
                 use_dispersive=False, use_dispersive_block=False):
        # 不调用 super().__init__()，而是手动构建
        # 因为需要用 WaveMST_ML 替代 WaveMST_3D 作为 prior
        nn.Module.__init__(self)
        
        if num_blocks is None:
            num_blocks = [2, 2, 2]
        self.num_stages = num_stages
        self.share_weights = share_weights
        self.nC = dim
        self.size = size
        self.len_shift = len_shift
        self.use_dispersive = use_dispersive
        self.use_source_injection = False  # stage4 不再支持源注入
        
        # ParaEstimator
        self.rho_estimators = nn.ModuleList([
            ParaEstimator(in_nc=dim) for _ in range(num_stages)
        ])
        
        # Prior: WaveMST_ML（替代 WaveMST_3D）
        from wpo3d import WaveMST_ML
        if share_weights:
            self.shared_prior = WaveMST_ML(
                dim=dim, stage=stage, num_blocks=num_blocks,
                ml_type=ml_type, unet_mode=unet_mode,
                mask_mode=mask_mode, use_kg=use_kg,
                use_dispersive_block=use_dispersive_block,
            )
            self.priors = None
        else:
            self.priors = nn.ModuleList([
                WaveMST_ML(
                    dim=dim, stage=stage, num_blocks=num_blocks,
                    ml_type=ml_type, unet_mode=unet_mode,
                    mask_mode=mask_mode, use_kg=use_kg,
                    use_dispersive_block=use_dispersive_block,
                )
                for _ in range(num_stages)
            ])
            self.shared_prior = None
        
        self.initial_conv = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        
        if use_dispersive:
            from enhancement_ops import DispersionCorrector
            self.dispersion_corrs = nn.ModuleList([
                DispersionCorrector(dim) for _ in range(num_stages)
            ])
    
    # get_prior 和 forward 直接继承自 WaveMST_3D_Unfold（不需要重写）
```

**关键**：`forward` 方法从 `WaveMST_3D_Unfold` 继承，其中调用 `self.get_prior(k)(z, Phi)` 会自动用 `WaveMST_ML` 而非 `WaveMST_3D`，因为我们在 `__init__` 中把 prior 换了。

但要**注意一个问题**：因为我们跳过了 `super().__init__()` 而直接调用 `nn.Module.__init__()`，需要确保 `forward` 方法能正常工作。最安全的做法是**直接复制 `WaveMST_3D_Unfold.forward` 的代码到这个类中**，去掉源注入相关的分支。

```python
    def get_prior(self, k):
        return self.shared_prior if self.share_weights else self.priors[k]
    
    def forward(self, g, input_mask):
        """复制自 WaveMST_3D_Unfold.forward，去掉源注入"""
        Phi, PhiPhiT = input_mask
        Phi_shift = shift_batch(Phi, self.len_shift)
        
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)
        f0 = shift_back_batch(temp_g, self.len_shift, self.size)
        f = self.initial_conv(torch.cat([f0, Phi], dim=1))
        
        outputs = []
        for k in range(self.num_stages):
            rho_k = self.rho_estimators[k](f)
            Phi_f = mul_Phi_f(Phi_shift, f, self.len_shift)
            residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
            residual = residual.clamp(min=-10, max=10)
            z = f + rho_k * mul_PhiT_residual(
                Phi_shift, residual, self.len_shift, self.size
            )
            
            f = self.get_prior(k)(z, Phi)
            
            if self.use_dispersive:
                f = self.dispersion_corrs[k](f)
            
            outputs.append(f)
        
        return outputs
```

---

## 5. 修改文件：train.py

### 5.1 新的 CONFIG 区

删除源注入相关选项，新增 ML 和 U-Net 选项：

```python
# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════
MODEL_INDEX  = 0       # 见 MODELS 字典
GPU_ID       = '0'
BATCH_SIZE   = 2
MAX_EPOCH    = 300
LR           = 4e-4
SCHEDULER    = 'CosineAnnealingLR'
EPOCH_SAMPLE = 5000
CROP_SIZE    = 256
NUM_BANDS    = 28
DIM          = 28
STAGE        = 3       # U-Net 内部 encoder 层数（传给 WaveMST）
NUM_BLOCKS   = [2, 2, 2]
MASK_MODE    = 'A'     # 'A' 纯 WPO，'D' KG（论文主推用 'A'，KG 用 MODEL_INDEX 切）
SAVE_THRESH  = 28.0

# ── ML 层选择 ──
ML_TYPE      = 'dwconv_ca'    # 'dwconv_ca' / 'wssa' / 'wave_aware'

# ── U-Net 骨架选择 ──
UNET_MODE    = 'symmetric'    # 'symmetric' / 'asymmetric' / 'alternating'

# ── Unfolding 配置（仅 IS_UNFOLDING=True 时生效）──
IS_UNFOLDING        = True
NUM_STAGES          = 5
SHARE_STAGE_WEIGHTS = True
MULTI_STAGE_LOSS    = True

# ── 色散介质（保留，待测试）──
USE_DISPERSIVE       = False
USE_DISPERSIVE_BLOCK = False

# ── 时空优化 ──
USE_AMP       = False
CACHE_PHIPHIT = True
# ════════════════════════════════════════════
```

### 5.2 新的 MODELS 字典

```python
MODELS = {
    # 端到端（无 unfolding）
    0: ('WaveMST_ML',        'ml_wpo_e2e'),          # ML+WPO 端到端
    1: ('WaveMST_ML_KG',     'ml_kg_e2e'),           # ML+KG  端到端
    # Unfolding
    7: ('WaveMST_ML_Unfold', 'ml_wpo_unfold'),       # ML+WPO unfolding
    8: ('WaveMST_ML_KG_Unfold', 'ml_kg_unfold'),     # ML+KG  unfolding
}
```

### 5.3 新的 build_model

```python
def build_model(index):
    use_kg = index in [1, 8]
    
    if index in [0, 1]:
        # 端到端
        from wpo3d import WaveMST_ML
        return WaveMST_ML(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            ml_type=ML_TYPE, unet_mode=UNET_MODE,
            mask_mode=MASK_MODE, use_kg=use_kg,
            use_dispersive_block=USE_DISPERSIVE_BLOCK,
        )
    elif index in [7, 8]:
        # Unfolding
        from wpo3d_unfold import WaveMST_ML_Unfold
        return WaveMST_ML_Unfold(
            dim=DIM, stage=STAGE, num_blocks=NUM_BLOCKS,
            num_stages=NUM_STAGES, share_weights=SHARE_STAGE_WEIGHTS,
            ml_type=ML_TYPE, unet_mode=UNET_MODE,
            mask_mode=MASK_MODE, use_kg=use_kg,
            size=CROP_SIZE, len_shift=2,
            use_dispersive=USE_DISPERSIVE,
            use_dispersive_block=USE_DISPERSIVE_BLOCK,
        )
    raise ValueError(f"无效 MODEL_INDEX: {index}")
```

### 5.4 Print 当前组合

在 `main()` 中模型创建后，加入详细的组合信息打印：

```python
# 在 model = build_model(MODEL_INDEX) 之后

print("=" * 60)
print(f"当前配置组合:")
print(f"  模型:     {MODELS[MODEL_INDEX][0]}")
print(f"  ML 层:    {ML_TYPE}")
print(f"  U-Net:    {UNET_MODE}")
print(f"  KG方程:   {'是' if MODEL_INDEX in [1, 8] else '否'}")
if IS_UNFOLDING:
    print(f"  展开:     {NUM_STAGES} stage, "
          f"{'共享权重' if SHARE_STAGE_WEIGHTS else '独立权重'}")
    print(f"  多阶段损失: {'是' if MULTI_STAGE_LOSS else '否'}")
else:
    print(f"  展开:     无 (端到端)")
print(f"  色散介质:  block={'是' if USE_DISPERSIVE_BLOCK else '否'}, "
      f"stage={'是' if USE_DISPERSIVE else '否'}")
print(f"  参数量:   {count_params(model):.2f}M")
print("=" * 60)
```

**输出示例**：

```
============================================================
当前配置组合:
  模型:     WaveMST_ML_Unfold
  ML 层:    wave_aware
  U-Net:    symmetric
  KG方程:   否
  展开:     5 stage, 共享权重
  多阶段损失: 是
  色散介质:  block=否, stage=否
  参数量:   0.95M
============================================================
```

### 5.5 train.py 训练循环

训练循环**不需要修改**——因为 `WaveMST_ML` 和 `WaveMST_3D` 的 forward 签名完全相同，`WaveMST_ML_Unfold` 和 `WaveMST_3D_Unfold` 的 forward 签名也完全相同。所有数据流路径不变。

---

## 6. 关键陷阱与调试

### 6.1 ML_WSSA 的窗大小与图像大小

`ML_WSSA` 的 `window_size=8`，要求 H 和 W 都能被 8 整除。CROP_SIZE=256 满足（256/8=32）。

但如果 U-Net 下采样到深层（dim_stage=4C），空间尺寸变为 64×64，仍可被 8 整除。再下一层 32×32 也行。但如果 STAGE=3，bottleneck 空间尺寸 = 256/8 = 32，可整除。

**潜在问题**：如果 CROP_SIZE 不是 8 的倍数（如 384），某些层可能会出问题。但 256 是安全的。

### 6.2 ML_WaveAware 的 FFT 开销

`ML_WaveAware` 内部做一次 2D rFFT（仅用于统计频谱能量）。rFFT 本身很快（256×256 约 0.1ms），但在 U-Net 每层每个 block 都做一次：

block 总数 = num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[2] + num_blocks[1] + num_blocks[0] = 2+2+2+2+2+2 = 12

12 次 rFFT ≈ 1.2ms per forward pass。对比 WPO 的 3D rFFT（每次约 5ms），增加约 20%——可接受。

### 6.3 ML_WPO_Block 比 WPO3DBlock 多用的显存

ML 层引入了额外的中间激活。估算：

- ML_DWConvCA：+1 份 [B, C, H, W]（DWConv 输出）
- ML_WSSA：+3 份 [B, C, H, W]（Q, K, V）
- ML_WaveAware：+1 份 [B, C, H, W] + FFT 结果

shared weights + ML_DWConvCA：显存约增加 15%
shared weights + ML_WSSA：显存约增加 30%

如果 BATCH_SIZE=2 在 stage2 正常运行（约 18GB），stage4 用 ML_DWConvCA 大约 21GB，用 ML_WSSA 大约 23GB——单 RTX 3090（24GB）仍可运行。

### 6.4 确保 WPO3D 不被修改

在 `ML_WPO_Block` 中，`self.wpo = WPO3D(dim, ...)` 是实例化了一个新的 WPO3D。它和原来的 `WPO3DBlock` 里的 WPO3D 是完全相同的类——只是多了一层 ML 包装。**原始 WPO3D 类本身一行不改。**

### 6.5 checkpoint 兼容性

stage4 的 checkpoint 与 stage2 **不兼容**（多了 ML 层的参数）。不要尝试加载 stage2 的 checkpoint 到 stage4 模型。

---

## 7. 开发与验证顺序

### 7.1 推荐顺序

**：ml_layers.py**

- [ ] 实现三个 ML 层类
- [ ] 单元测试：随机输入 [B=2, C=28, H=256, W=256]，forward 不报错，输出 shape 正确
- [ ] 每个 ML 层 backward 不产生 NaN

**：wpo3d.py 修改**

- [ ] 新增 `ML_WPO_Block` 和 `WaveMST_ML`
- [ ] 测试端到端模型 `WaveMST_ML`：forward + backward 跑通
- [ ] 对比参数量：`WaveMST_ML(ml_type='dwconv_ca')` vs `WaveMST_3D`

**：train.py + 端到端快速验证**

- [ ] 更新 CONFIG 和 MODELS
- [ ] **先跑端到端（MODEL_INDEX=0, IS_UNFOLDING=False）**
- [ ] 用 ML_TYPE='dwconv_ca'（最轻量）跑 100 epoch
- [ ] **验收标准：PSNR > 35.0 dB**（vs 纯 WPO 端到端 34.70 dB）
- [ ] 如果 > 35.0：ML 层有效，继续
- [ ] 如果 < 34.5：ML 层有问题，检查残差连接

**：wpo3d_unfold.py + unfolding 验证**

- [ ] 新增 `WaveMST_ML_Unfold`
- [ ] 用最佳 ML_TYPE + IS_UNFOLDING=True，NUM_STAGES=5，跑 100 epoch
- [ ] **验收标准：PSNR > 38.5 dB**（vs 纯 WPO 5stg 38.21 dB）

**：消融实验**

- [ ] 三种 ML_TYPE 各跑一次端到端（100 epoch），选最佳
- [ ] 最佳 ML_TYPE + 三种 UNET_MODE 各跑一次端到端
- [ ] 最佳组合 + unfolding 跑 300 epoch
- [ ] 可选：加 USE_DISPERSIVE_BLOCK=True 测试色散

### 7.2 验证检查表

| 阶段 | 指标 | 下限 | 说明 |
|------|------|------|------|
| 端到端 @30 epoch | PSNR | > 32.0 | 说明在学 |
| 端到端 @100 epoch | PSNR | > 35.0 | ML 层有效 |
| Unfolding @30 epoch | PSNR | > 34.0 | 说明在学 |
| Unfolding @100 epoch | PSNR | > 38.5 | 超过纯 WPO 5stg |
| Unfolding @300 epoch | PSNR | > 39.0 | 目标 |