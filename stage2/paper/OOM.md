# WaveMST: 基于波动方程的高光谱压缩感知重建

## 组会 PPT 详细内容

---

# 第一部分：问题背景

## 1.1 CASSI 系统与成像过程

高光谱成像（Hyperspectral Imaging, HSI）旨在获取场景在密集光谱波段上的空间-光谱三维数据立方体 $f(x, y, \lambda)$。传统的逐波段扫描方式需要多次曝光，在动态场景中不可行。编码孔径快照光谱成像系统（Coded Aperture Snapshot Spectral Imaging, CASSI）通过编码孔径和色散元件的巧妙组合，将三维光谱信息压缩到一幅二维测量图像中，实现了单次曝光获取高光谱数据的能力。

**CASSI 前向模型：**

$$g(x, y) = \sum_{c=1}^{C} \Phi_c(x, y) \cdot f(x, y - d_c, \lambda_c) + noise$$

其中：
- $f \in \mathbb{R}^{H \times W \times C}$：原始高光谱立方体（28 波段，波长 453–681 nm）
- $\Phi_c(x, y)$：编码孔径 mask 在波段 c 的采样模式（二值 0/1）
- $d_c = c \times \text{len\_shift}$：色散位移（棱镜导致不同波长向不同方向偏移）
- $g \in \mathbb{R}^{H \times W'}$：最终的二维压缩测量（$W' = W + (C-1) \times \text{len\_shift}$）

**压缩比**：从一幅 $256 \times 310$ 的二维测量重建 $256 \times 256 \times 28$ 的三维立方体，压缩比约为 **28:1**。这是一个严重欠定（ill-posed）的逆问题。

**Shift 操作的物理意义**：棱镜色散导致不同波长的光在探测器上落在不同空间位置。波长越长，偏移越大。我们的实现中 `len_shift=2`，即相邻波段偏移 2 个像素。

---

## 1.2 现有方法的分类与局限

### 基于优化的方法
- TwIST、GAP-TV、DeSCI
- 手工设计正则项（全变差 TV、稀疏性、低秩）
- 优点：有数学保证（收敛性）
- 缺点：先验太弱，迭代慢（几百次迭代），PSNR 约 30–33 dB

### 端到端深度学习方法
- TSA-Net (ECCV'20)、HDNet (CVPR'22)、MST/MST++ (CVPR'22)、CST (ECCV'22)
- 直接学一个 $g \to f$ 的映射
- 优点：速度快，PSNR 高
- 缺点：**完全是黑箱**——为什么用 self-attention？为什么这样设计 token mixing？缺乏物理动机

### 深度展开方法（Deep Unfolding）
- ADMM-Net、DGSMP (CVPR'21)、GAP-Net、RDLUF (ECCV'22)、PADUT (CVPR'23)
- 将优化算法展开为 K 个 stage：$f^{(k+1)} = \text{Prior}(f^{(k)} + \rho_k \Phi^T \frac{g - \Phi f^{(k)}}{\Phi\Phi^T})$
- 优点：可解释的迭代结构
- **但 Prior 网络仍然是任意的 CNN/Transformer，缺乏物理约束**

### 核心空白
> 所有现有方法都忽略了一个根本事实：**高光谱图像的每个波段对应一个确定的物理波长**。453 nm 蓝光和 681 nm 红光有本质的物理差异——波数不同、衍射极限不同、在介质中的传播特性不同。这些物理知识完全可以作为归纳偏置（inductive bias）注入网络设计中。

---

## 1.3 本工作的动机与核心思路

**核心观察**：高光谱数据的光谱维度 $\lambda$ 本质上是电磁波的波长。既然我们处理的数据来源于物理波动，那么网络本身的 token mixing 操作就应该遵循波动方程的传播规律。

**类比**：
| 传统方法 | 本工作 |
|---------|--------|
| Self-Attention: $\text{softmax}(QK^T/\sqrt{d})V$ | WPO3D: 波动方程频域闭式解 |
| 位置编码 (PE) | 物理波长编码 $k_{\text{phys}} = \lambda_{\min}/\lambda_b$ |
| 可学习温度参数 | 可学习阻尼系数 $\alpha$、波速 $v_s, v_l$ |
| 注意力 mask | 编码孔径作为初始条件振幅门控 |

**一句话总结**：我们将 CASSI 重建建模为**各向异性阻尼波在非均匀色散介质中的传播问题**，波动方程的频域闭式解天然取代了 self-attention 作为全局 token mixing 算子。

---

# 第二部分：物理与数学基础

## 2.1 电磁波与波动方程

光是电磁波。Maxwell 方程组在均匀各向同性介质中可简化为标量波动方程：

$$\frac{\partial^2 u}{\partial t^2} = v^2 \nabla^2 u$$

其中 $v$ 是传播速度（由介质的折射率 $n$ 决定：$v = c/n$），$\nabla^2$ 是 Laplacian 算子。

对于高光谱数据，我们引入两个关键扩展：

1. **各向异性**：空间维度 $(x, y)$ 和光谱维度 $\lambda$ 的"传播速度"不同
2. **阻尼**：加入衰减项，控制传播距离

得到我们使用的**各向异性阻尼波动方程**：

$$\frac{\partial^2 u}{\partial t^2} + \alpha \frac{\partial u}{\partial t} = v_s^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + v_l^2 \frac{\partial^2 u}{\partial \lambda^2}$$

其中：
- $\alpha > 0$：阻尼系数（控制信息传播的衰减速率）
- $v_s$：空间波速（控制空间维度的信息混合范围）
- $v_l$：光谱波速（控制光谱维度的信息混合范围）

## 2.2 频域闭式解

对上述 PDE 做 3D Fourier 变换（空间 + 光谱），利用初始条件 $u(0) = u_0$, $\dot{u}(0) = v_0$，得到频域闭式解：

$$\hat{u}(k, t) = e^{-\frac{\alpha t}{2}} \left[ \hat{u}_0 \cdot C(k, t) + \left(\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0\right) \cdot S(k, t) \right]$$

其中色散关系为：
$$\omega^2(k) = v_s^2(k_x^2 + k_y^2) + v_l^2 k_\lambda^2$$

判别量 $\eta = \omega^2 - (\alpha/2)^2$ 决定传播行为：

| $\eta > 0$（欠阻尼） | $\eta < 0$（过阻尼） |
|---|---|
| $C = \cos(\omega_d t)$ | $C = \cosh(\gamma t)$ |
| $S = \sin(\omega_d t) / \omega_d$ | $S = \sinh(\gamma t) / \gamma$ |
| 低频模态 → 振荡传播 | 高频模态 → 指数衰减 |

**物理直觉**：低频信息（大尺度结构）以振荡方式长距离传播；高频信息（细节纹理）快速衰减，仅局部混合。这天然构成了一个**自适应频率滤波器**。

## 2.3 傅里叶变换的优势与不确定性原理

**为什么用 FFT 而非空间域卷积？**

1. **全局感受野**：FFT 天然覆盖全部空间频率，无需堆叠多层
2. **O(N log N) 复杂度**：相比 self-attention 的 O(N²)，对 $256 \times 256 \times 28$ 数据高效得多
3. **物理自洽**：波动方程本身在频域有闭式解，不是人为选择

**Heisenberg 不确定性原理**：
$$\Delta x \cdot \Delta k \geq \frac{1}{2}$$

频域分辨率越高（$\Delta k$ 越小），空间定位越模糊（$\Delta x$ 越大）。

**我们如何解决**：
- WPO3D 的**阻尼机制**天然实现了频率-空间的 trade-off：高频模态被阻尼快速衰减（等效于有限空间范围），低频模态长距离传播（全局结构）
- **Mask 门控**（MaskGateA）提供空间先验：mask 值为 0 的位置振幅被衰减，相当于告诉网络"这些位置缺失了信息"
- U-Net 的**多尺度结构**进一步缓解：下采样后做 FFT，等效于在不同分辨率级别处理不同频带

## 2.4 FFT 的实现优化：Pad to 2^n

cuFFT 对长度为 $2^n$ 的序列最高效。我们的光谱维度 $C=28$，不是 2 的幂。

解决方案：FFT 前将光谱维度 pad 到 32（$2^5$），iFFT 后截取前 28 通道：
```
C=28 → pad → C_fft=32 → FFT → wave modulate → iFFT → truncate → C=28
```

实测加速约 15–20%，精度无损失（零填充不引入频谱泄露）。

---

# 第三部分：网络架构设计

## 3.1 WPO3D — 波动传播算子（核心模块）

WPO3D 是网络的核心 token mixing 操作，替代 self-attention：

```
输入 x [B, C, H, W] + mask_spatial [B, C, H, W]
    │
    ├── MaskGateA: gate = ε + (1-ε)·mask
    │     u0 = Φ(x) · gate     ← 初始位移场
    │     v0 = Ψ(x) · gate     ← 初始速度场
    │
    ├── 3D rFFT (pad C→32)
    │
    ├── 频域波动调制
    │     ω² = vs²(kh² + kw²) + vl²·kc²
    │     η = ω² − (α/2)²
    │     out_fft = decay · [u0·C(η,t) + (v0 + α/2·u0)·S(η,t)]
    │
    ├── 3D irFFT (truncate →C=28)
    │
    ├── [可选] 色散修正: f + w·δv(r)·∇²f
    │
    ├── LayerNorm → SiLU gate → Linear projection
    │
    └── 输出 [B, C, H, W]
```

**可学习参数（仅 4 个标量 + 门控网络）**：
| 参数 | 物理含义 | 初始化 | 约束 |
|------|---------|--------|------|
| $\alpha$ | 阻尼系数 | 0.1 | softplus → 正值 |
| $v_s$ | 空间波速 | 1.0 | softplus → 正值 |
| $v_l$ | 光谱波速 | 0.5 | softplus → 正值 |
| $t$ | 传播时间 | 1.0 | softplus → 正值 |

**Mask 的物理角色**：CASSI 的 coded aperture 在光学系统中对光场做了一次振幅调制。我们将其建模为波动方程的**初始条件软门控**——mask=0 的位置光被遮挡，初始振幅为 ε（不完全为零，保留梯度流）。

## 3.2 WPO3D Block

标准 Pre-Norm 残差块：
```
x → LN → WPO3D(·, mask) → + → LN → FFN → + → out
│                          ↑   │              ↑
└──────────────────────────┘   └──────────────┘
```

FFN 结构：Conv1×1 → GELU → DWConv3×3 → GELU → Conv1×1（参照 WaveFormer 的 Mlp）。

## 3.3 Wave-3D 整体架构（U-Net 骨架）

参照 MST 的 U-Net 设计，但将 S-MSA 完全替换为 WPO3D Block：

```
输入 x [B, 28, H, W]
    │
    Conv3×3 Embedding → [B, dim, H, W]
    │
    ┌─ Encoder Stage 1: 2× WPO3DBlock(dim) → downsample(×2) → dim*2
    │    └─ mask_down: Conv4×4 + sigmoid（mask 同步下采样）
    │
    ├─ Encoder Stage 2: 2× WPO3DBlock(dim*2) → downsample(×2) → dim*4
    │
    ├─ Bottleneck: 2× WPO3DBlock(dim*4)
    │
    ├─ Decoder Stage 2: upsample → fusion(cat) → 2× WPO3DBlock(dim*2)
    │
    └─ Decoder Stage 1: upsample → fusion(cat) → 2× WPO3DBlock(dim)
    │
    Conv3×3 Mapping → [B, 28, H, W] + 残差连接(x)
```

**mask 在多尺度中的传播**：mask 通过可学习的 Conv4×4 下采样（+ sigmoid 归一化到 [0,1]），在每个分辨率级别为 WPO3D 提供空间先验。Decoder 中直接复用 Encoder 对应层的 mask（skip 连接）。

## 3.4 Deep Unfolding（GAP 框架）

将优化迭代展开为 K 个可学习的 stage，每个 stage 包含：

$$f^{(k+1)} = \text{Wave-3D}\left(f^{(k)} + \rho_k \cdot \Phi^T \frac{g - \Phi f^{(k)}}{\Phi\Phi^T}\right)$$

```
测量 g [B, 1, H, W']
    │
    初始化: shift_back(g/C×2) → concat(Phi) → Conv1×1 → f₀
    │
    ┌── Stage 1 ──────────────────────────────────────┐
    │  GD step:                                       │
    │    residual = (g − Φf) / ΦΦᵀ                   │
    │    z = f + ρ₁ · Φᵀ(residual)                   │
    │  [源项注入]: z = Conv([z ‖ Φᵀg])               │
    │  Prior:                                         │
    │    f = Wave-3D(z, Phi)  ← 内含色散修正      │
    └─────────────────────────────────────────────────┘
    │
    ┌── Stage 2 ─── ... ─── Stage K ──┐
    │        (结构相同，权重可共享)       │
    └──────────────────────────────────┘
    │
    输出: [f₁, f₂, ..., fₖ]（多 stage 损失监督）
```

**关键设计**：
- **ParaEstimator**：AdaptiveAvgPool → MLP → $\rho_k$，每个 stage 独立预测步长
- **PhiPhiT 预缓存**：mask 固定，$\Phi\Phi^T$ 仅需计算一次（无代价优化）
- **Shift 向量化**：scatter/gather 替代逐通道 for 循环（28 次 kernel launch → 1 次）
- **多 stage 加权损失**：$\mathcal{L} = \mathcal{L}_K + \sum_{k<K} w_k \mathcal{L}_k$（DPU 风格，加速收敛）

## 3.5 WPO3D 内置增强：空间色散修正（或者放到gap里面每stage也行 都是正交的）

**动机**：WPO3D 的色散关系 $\omega^2 = v_s^2 k_s^2 + v_l^2 k_\lambda^2$ 假设**全局均匀介质**（所有空间位置共享同一组波速参数）。但真实场景中不同区域的光学性质不同（如边缘 vs. 平坦区域）。

**方案**：Born 近似一阶修正——在 WPO3D 内部（iFFT 回到空间域后、输出投影前），学习空间依赖的速度扰动 $\delta v(x, y)$：

$$f_{\text{out}} = f + w \cdot \delta v(x, y) \cdot \nabla^2 f$$

**在 WPO3D 中的位置**：
```
u0, v0 → 3D FFT → 频域波动调制 → 3D iFFT → 【色散修正】→ LayerNorm → gate → proj
```

其中：
- $\delta v(x, y)$：由轻量 DWConv→ReLU→Conv1×1→Tanh 网络从 f 预测（空间场）
- $\nabla^2 f$：固定 Laplacian 核（$[0,1,0; 1,-4,1; 0,1,0]$），reflect padding，不参与训练
- $w$：可学习标量，初始化 0.1（弱修正起步，避免训练初期不稳定）

**物理含义**：波在均匀介质中传播后（iFFT 完成），立即经历一次空间非均匀介质的散射修正。这对应光学中的"弱散射 Born 近似"——假设非均匀性很小（$\delta v \in [-1,1]$），只取一阶修正。

**与 WPO3D 内置色散关系的区别**：
| | WPO3D 频域色散 | DispersionCorrector |
|---|---|---|
| 建模维度 | 频率维度：不同 $k$ 传播速度不同 | 空间维度：不同 $(x,y)$ 波速不同 |
| 数学操作 | 频域乘法（全局参数） | 空间域卷积（逐像素） |
| 参数 | $v_s, v_l$（标量） | $\delta v(x,y)$（空间场，输入依赖） |
| 物理类比 | 均匀各向异性晶体中的色散 | 有杂质/缺陷的非均匀介质 |

两者正交，互不冲突——先做频率色散传播，再做空间非均匀修正。

## 3.6 Unfolding 增强：源项注入（Source Injection）

**放置位置**：在 GAP unfolding 循环中，每个 stage 的 GD step 之后、Prior 之前。

**动机**：Deep unfolding 多次迭代后，原始测量信息 $g$ 逐渐被遗忘（信息衰减问题）。GD step 虽然每次都用到 $g$，但经过 $\Phi^T$ 反投影和步长 $\rho_k$ 缩放后，原始测量的"指纹"逐渐模糊。

**方案**：预计算 $\Phi^T g$（测量值的一次性反投影），在每个 stage 的 prior 之前注入：

$$z_{\text{enhanced}} = \text{Conv}_{1\times1}([z \| \Phi^T g])$$

其中 $[·\|·]$ 为 channel 拼接，Conv 将 2C→C 通道融合。

**在 Unfolding 循环中的位置**：
```
┌── Stage k ─────────────────────────────────────────┐
│  GD step:                                          │
│    residual = (g − Φf) / ΦΦᵀ                      │
│    z = f + ρₖ · Φᵀ(residual)                      │
│                                                    │
│  【源项注入】: z = Conv([z ‖ Φᵀg])                  │
│                                                    │
│  Prior: f = WaveMST-3D(z, Phi)                     │
└────────────────────────────────────────────────────┘
```

**物理含义**：类比波动力学中的**持续激励源**——每个迭代步都能直接看到原始测量的反投影，不只依赖上一步的估计。相当于在每一轮波传播前都重新"点亮"原始信号。

**实现细节**：
- $\Phi^T g$ 只算一次（mask 和 g 都固定），存储开销为 [B, C, H, W]
- 每个 stage 用独立的 Conv1×1 融合（允许各 stage 学习不同的注入强度）
- 与 GD step 的 $\Phi^T r$ 不同：GD step 注入的是残差的反投影（随迭代变化），源项注入的是测量值本身的反投影（固定锚点）

---

# 第四部分：训练策略

## 4.1 损失函数

- **主损失**：RMSE $= \sqrt{\text{MSE}(f_{\text{pred}}, f_{\text{gt}})}$
- **多 stage 加权**：最后一个 stage 权重最大，前面 stage 递减权重辅助收敛

## 4.2 训练配置

| 配置项 | 值 |
|--------|-----|
| Optimizer | Adam, lr=4e-4 |
| Scheduler | CosineAnnealingLR, T_max=300 |
| Batch size | 2（受显存限制） |
| Crop size | 256×256 |
| Epochs | 300 |
| GPU | RTX3090 |

## 4.3 数据集

- **训练集**：CAVE 1024×1024, 28 波段，随机 crop + 旋转/翻转增强
- **测试集**：TSA 模拟数据，10 个标准场景

## 4.4 混合精度训练（AMP）

`torch.cuda.amp.autocast` + `GradScaler`，降显存约 30%，加速约 20%，精度损失可忽略（< 0.05 dB）。

---

# 第五部分：实验结果

## 5.1 当前结果

| 配置(300epoch) | PSNR (dB) | 备注 | 参数量 |
|------|-----------|------|---|
| wpo3d+maskA | 34.7 | 软门控| 0.79M |
| wpo3d+other masks| 33.8—34.69 | 其他可行的mask方式 | 0.79M |
| wpo3d+maskA+色散介质| 35.6 | 内部加色散介质 | 0.80M|
| 5-stage, share_weights，无源注入和色散 | **37.7** | 100 epoch，但是服务器断电了 | ~0.82M |
| 预估完全体 (9-stage, not share, 300 epoch + enhancements) | ~39.5–40.5 | 估计值 | ~4.0M |

## 5.2 Stage-wise PSNR 演化

Unfolding 的优势：可以观察每个 stage 的重建质量如何递增

```

```

## 5.3 计算效率对比（卖点）

| 方法 | 参数量 | 复杂度 | 物理可解释 |
|------|--------|--------|-----------|
| MST++ | ~1.3M | O(N²) attention | ✗ |
| CST | ~3M | O(N²) attention | ✗ |
| PADUT | ~5.4M | O(N²) attention | ✗ |
| **WaveMST (ours, share)** | **~0.8M** | **O(N log N) FFT** | **✓** |
| WaveMST (ours, full) | ~4M | O(N log N) FFT | ✓ |

**关键图表建议**：画一张 **PSNR vs. Params** 散点图，WaveMST 在 Pareto 前沿。

## 5.4 与 SOTA 的对比叙事

> "With 5× fewer parameters and physically grounded design, our method achieves competitive performance while providing interpretable intermediate representations that directly correspond to wave propagation states."

不是"我们比 SOTA 差一点"，而是"以极低的参数量和清晰的物理语义达到了接近 SOTA 的性能，且每个模块都有明确的物理对应"。

---

# 第六部分：物理可解释性分析

## 6.1 学习到的物理参数含义

训练后可以可视化每层学到的 $(α, v_s, v_l, t)$：
- 浅层（高分辨率）：$v_s$ 大（空间方向长距离传播）、$\alpha$ 小（少阻尼）
- 深层（低分辨率，通道多）：$v_l$ 相对增大（光谱方向混合增强）
- 预期行为：底层提取空间结构，深层融合光谱信息

## 6.2 阻尼的自适应频率选择

$\eta = \omega^2 - (\alpha/2)^2$ 划分了两个频率区域：
- $\eta > 0$（低频）：cos/sin 振荡 → 信息长距离传播 → 提取全局结构
- $\eta < 0$（高频）：cosh/sinh + 衰减 → 局部混合 → 保留细节

这是一个**物理驱动的自适应频率滤波器**，不需要手动设计截止频率。

## 6.3 Mask 作为初始条件的物理意义

CASSI 的 coded aperture 在光学上是对光场的一次性振幅调制。在波动方程框架下：
- mask=1 的位置：光场完整通过，初始振幅保持
- mask=0 的位置：光被遮挡，初始振幅衰减到 ε=0.1

网络学习到的是：如何从这些"部分遮挡的初始条件"出发，通过波传播来"填充"被遮挡的区域——物理上这就是波的衍射和干涉现象。

---

# 第七部分：总结与展望

## 7.1 核心贡献

1. **WPO3D**：首次将 3D 阻尼波动方程的频域闭式解作为 token mixing 算子，O(N log N) 复杂度，4 个可学习物理参数
2. **物理编码的 Mask 机制**：将 CASSI 的 coded aperture 自然嵌入波动方程初始条件，而非后处理
3. **GAP Unfolding + WPO Prior**：数据保真（CASSI 测量模型）+ 物理先验（波动方程）的双物理驱动
4. **空间色散修正**：Born 近似一阶修正，将均匀介质假设推广到非均匀介质
5. **源项注入**：解决 deep unfolding 中的信息衰减问题

## 7.2 局限与未来方向

- **空间自适应波速编码器**：将全局标量 $v_s$ 升级为空间场 $v_s(x, y)$
- **物理波长注入**：目前未直接使用 453–681 nm 的物理波长信息作为频率编码
- **更多 stage + 更大模型**：9-stage 不共享权重，预计可达 40+ dB

## 7.3 为什么这个工作有意义（One More Thing）

这不只是"又一个 CASSI 重建网络"。我们展示了一种**范式**：

> 如果你的数据来源于物理过程（波动、扩散、输运），那么网络的 token mixing 操作就应该直接由对应的 PDE 闭式解来实现，而不是用通用的 attention 或卷积去拟合。

WPO3D 对波动方程的实现，可以自然推广到：
- 医学超声成像重建
- 地震波层析成像
- 声学信号处理
- 任何可以用 PDE 建模的逆问题

---

# 附录：关键公式速查

**色散关系**：
$$\omega^2 = (2\pi)^2 \left[ v_s^2(f_h^2 + f_w^2) + v_l^2 f_c^2 \right]$$

**频域波动调制**：
$$\hat{u}(t) = e^{-\alpha t/2} \left[ \hat{u}_0 \cdot C(\eta, t) + \left(\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0\right) \cdot S(\eta, t) \right]$$

**GAP 梯度下降步**：
$$z^{(k)} = f^{(k)} + \rho_k \cdot \Phi^T \frac{g - \Phi f^{(k)}}{\Phi\Phi^T}$$

**Born 色散修正**：
$$f_{\text{out}} = f + w \cdot \delta v(x,y) \cdot \nabla^2 f$$

**CASSI 前向模型**（矩阵形式）：
$$g = \Phi f + n, \quad \Phi = \text{shift}(\text{diag}(\text{mask}))$$
