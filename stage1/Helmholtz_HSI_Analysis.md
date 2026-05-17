# 亥姆霍兹方程 × HSI × Mask：四个新方案的数学物理深度推导

> **文档定位**：本文是前两份文档（WaveMST_Analysis.md、WaveMST_Technical_Handoff.md）的延伸，专注于亥姆霍兹方程引入 HSI 的新方向。已在前文充分讨论的内容（3D WPO 闭式解、方案 A/B/C/D、路线 2/3 不等价性）不再重复。本文只写前文没有覆盖的真正新内容。
>
> **与前文的核心区别**：前文所有波速 $v_s, v_\lambda$ 是可学习参数（数据驱动）。本文引入**物理硬先验**：真实光波波数 $k(\lambda)=2\pi/\lambda$，把实验室可测量的物理常数直接编码进网络结构。

---

## 目录

1. [与前文的重复性精确分析](#1-与前文的重复性精确分析)
2. [物理基础：从麦克斯韦到亥姆霍兹](#2-物理基础从麦克斯韦到亥姆霍兹)
3. [方案一 H1-γ：亥姆霍兹频域逆算子 + Mask 源项调制](#3-方案一-h1-γ)
4. [方案二 H1-β：复折射率亥姆霍兹（吸收体建模）](#4-方案二-h1-β)
5. [方案三 H2-α精化：物理波数注入 WPO 色散关系](#5-方案三-h2-α精化)
6. [方案四 H2-γ：三合一统一框架](#6-方案四-h2-γ)
7. [四方案综合比较与推荐](#7-四方案综合比较与推荐)

---

## 1. 与前文的重复性精确分析

### 1.1 前文覆盖的内容（不重复）

- 3D 阻尼波动方程的完整闭式解推导（含欠阻尼/过阻尼统一形式）
- 路线 2 vs 路线 3 的数学不等价性证明
- 方案 A（初始振幅门控）：$u_0^M = [\epsilon+(1-\epsilon)M]\cdot u_0$
- 方案 B（源项 Duhamel 积分）：非齐次波动方程
- 方案 C（频域门控，证明等价于方案 A）
- 方案 D（Klein-Gordon 质量场 + Born 近似）
- Transformer 三条路线（纯 WPO / 替代 / 并联）

### 1.2 重复性精确定位

| 本文方案 | 与前文的关系 | 真正新增的内容 |
|---------|------------|-------------|
| H1-γ（源项调制） | 前文方案 B 是退化版（用波动 Green 函数，不含 $k^2$） | 亥姆霍兹算子 $1/(k^2-\|\boldsymbol{\omega}\|^2)$ 的频域逆结构是全新的；共振条件 $k(\lambda)=\|\boldsymbol{\omega}\|$ 在前文没有出现 |
| H1-β（复折射率） | 前文完全没有虚数折射率 | 全新：Beer-Lambert 吸收、复波数、Sommerfeld 辐射条件 |
| H2-α 精化 | 前文方案 A 是子集（无 $k_\text{phys}$） | 物理波数注入色散关系、软硬先验混合初始化公式 |
| H2-γ（三合一） | 前文无此组合 | 三步框架：初始门控 + 物理波数传播 + Beer-Lambert 吸收 |

### 1.3 前文的核心局限（本文要解决的）

前文所有方案的波速 $v_\lambda$ 对称处理光谱维度——假设波在光谱维度的传播与方向无关。但物理上 HSI 每个波段对应特定波长 $\lambda_b$（CAVE：400–700 nm，28 波段，步长约 10 nm），网络完全不知道第 1 通道是蓝光、第 28 通道是红光。这个物理身份信息被前文完全丢弃。

**本文的核心新意**：用 $k(\lambda)=2\pi/\lambda$ 把物理波长编码进网络，使不同波段有不同的固有振荡频率，符合电磁波的真实物理。

---

## 2. 物理基础：从麦克斯韦到亥姆霍兹

### 2.1 推导链

在线性、各向同性介质中，电场 $\mathbf{E}(\mathbf{r},t)$ 满足：

$$\nabla^2\mathbf{E} - \mu_0\epsilon(\mathbf{r})\frac{\partial^2\mathbf{E}}{\partial t^2}=0 \tag{2.1}$$

令 $\mathbf{E}(\mathbf{r},t)=\mathbf{E}(\mathbf{r})e^{-i\omega t}$（时谐分解），$\partial_t^2\to -\omega^2$：

$$\boxed{\nabla^2\mathbf{E}(\mathbf{r}) + k^2(\mathbf{r})\mathbf{E}(\mathbf{r})=0} \tag{2.2}$$

$$k(\mathbf{r})=\frac{\omega}{c}n(\mathbf{r})=\frac{2\pi}{\lambda}n(\mathbf{r}) \tag{2.3}$$

**关键**：亥姆霍兹方程是波动方程在**单频稳态**下的空间方程。

### 2.2 HSI 建模映射

| 光学物理量 | HSI 中的对应 | 是否已知 |
|----------|------------|--------|
| 电场 $\mathbf{E}(\mathbf{r})$ | 特征张量 $f(x,y,\lambda)$ | 网络计算 |
| 波长 $\lambda$ | 光谱波段 $\lambda_b\in\{400,\ldots,700\}$ nm | **已知** |
| 折射率 $n(\mathbf{r})$ | 地物光学响应（空间依赖） | 需学习 |
| 波数 $k(\lambda)$ | $k_b=2\pi/\lambda_b$ | **已知物理量** |
| 吸收系数 $\kappa$ | CASSI mask 低透射率区域 | **已知** |

### 2.3 CAVE 数据集的物理波数归一化

CAVE 28 个波段，$\lambda_b\in\{400,410,\ldots,670\}$ nm。

$$\tilde{k}_b = \frac{400}{\lambda_b}\in[0.597,\,1.0]$$

（归一化：以 400 nm 蓝光为参考，$\tilde{k}=1$；670 nm 红光 $\tilde{k}\approx0.597$。）

这个向量 $[\tilde{k}_1,\ldots,\tilde{k}_{28}]$ 是**免费的物理先验**，直接预计算，不需要学习。

### 2.4 各方程的关系图谱

```
麦克斯韦方程（时空）
      |
      | 时谐分解 e^{-iωt}
      v
亥姆霍兹：∇²f + k²f = 0            ← 本文出发点
      |
      | 加源项 -M·s
      v
非齐次亥姆霍兹                       ← H1-γ
      |
      | k → k+iκ（复折射率）
      v
复亥姆霍兹（吸收体）                  ← H1-β
      
亥姆霍兹 k² 注入 WPO 色散关系        ← H2-α
      |
      | 组合初始门控 + 吸收修正
      v
三合一框架                            ← H2-γ

[关系：亥姆霍兹 = 波动方程稳态极限；
 KG（前文方案D）= 亥姆霍兹时域形式（m² ↔ k²）]
```

---

## 3. 方案一 H1-γ：亥姆霍兹频域逆算子 + Mask 源项调制

### 3.1 方程设定

$$\nabla_{xy}^2 f(x,y,\lambda) + k^2(\lambda)\,f(x,y,\lambda) = -M(x,y)\cdot s(x,y,\lambda) \tag{3.1}$$

- $k(\lambda)=2\pi/\lambda_b$：已知物理波数（不可学习，或软化为可微调）
- $M(x,y)\in[0,1]$：CASSI mask（仅空间依赖）
- $s(x,y,\lambda)$：源场，由输入特征通过编码器预测
- $\nabla_{xy}^2$：仅对空间 $(x,y)$ 做 Laplacian，$\lambda$ 是参数

### 3.2 频域闭式解推导

对 (3.1) 两边做 2D 空间傅里叶变换 $\mathcal{F}_{xy}$。

利用 $\mathcal{F}[\nabla_{xy}^2 f]=-(ω_x^2+ω_y^2)\hat{f}$，令 $|\boldsymbol{\omega}|^2=\omega_x^2+\omega_y^2$：

$$-|\boldsymbol{\omega}|^2\hat{f}+k^2(\lambda)\hat{f}=-\widehat{Ms}(\boldsymbol{\omega},\lambda)$$

$$(k^2(\lambda)-|\boldsymbol{\omega}|^2)\hat{f}=-\widehat{Ms}$$

$$\boxed{\hat{f}(\boldsymbol{\omega},\lambda)=\frac{\widehat{Ms}(\boldsymbol{\omega},\lambda)}{k^2(\lambda)-|\boldsymbol{\omega}|^2+i\epsilon}} \tag{3.2}$$

其中 $\epsilon>0$ 是正则化参数（Sommerfeld 辐射条件）。

注意：$\widehat{Ms}=\mathcal{F}_{xy}[M\cdot s]$，在实现时直接空间域乘法再 FFT，不需要做 $\hat{M}*\hat{s}$ 的频域卷积，复杂度 $O(N\log N)$。

### 3.3 共振条件分析

分母 $k^2(\lambda)-|\boldsymbol{\omega}|^2=0$ 当 $|\boldsymbol{\omega}|=k(\lambda)=2\pi/\lambda$ 时成立。

这是**光学衍射极限**的数学表达：波长 $\lambda$ 的光能分辨的最小空间周期为 $\lambda$，对应空间频率 $|\boldsymbol{\omega}_\text{res}|=2\pi/\lambda$。

**在特征空间的含义**：

- 短波段（蓝光，$\lambda=400$ nm，$k$ 大）→ 共振空间频率高 → 该通道可携带高频空间信息（精细纹理）
- 长波段（红光，$\lambda=700$ nm，$k$ 小）→ 共振空间频率低 → 该通道只携带低频空间信息（大范围结构）

这恰好对应高光谱遥感的物理观测：近红外波段对大面积均匀地物（植被、水体）响应强，短波可见光对精细边界响应强。

### 3.4 正则化 $\epsilon$ 的设计

共振峰幅值：$|\hat{f}|_\text{res}=|\widehat{Ms}|/\epsilon$

峰宽（半高全宽）：$\Delta|\boldsymbol{\omega}|\approx\epsilon/(2|k|)$（通过分母展开得到）

$\epsilon$ 的两种设计：

**固定 $\epsilon$**：$\epsilon=0.01$，简单稳定。

**可学习 $\epsilon$**（推荐）：

$$\epsilon(\lambda)=\text{softplus}(\epsilon_\text{raw}(\lambda))+10^{-6}$$

每个波段有独立的 $\epsilon_b$，允许不同波段有不同的共振锐度。额外参数：28 个标量。

### 3.5 Mask 的作用机制

**低透射率区域** ($M\approx0$)：$Ms\approx0$，该位置无源项注入。但亥姆霍兹 Green 函数（Hankel 函数）会把邻近高透射率区域的信号传播过来：

$$f(x,y,\lambda)=\int G_k(\mathbf{r}-\mathbf{r}')M(\mathbf{r}')s(\mathbf{r}',\lambda)\,d\mathbf{r}'$$

$$G_k(|\mathbf{r}|)=\frac{i}{4}H_0^{(1)}(k|\mathbf{r}|)\approx\frac{e^{ik|\mathbf{r}|}}{\sqrt{|\mathbf{r}|}}\text{（远场柱面波）}$$

远场衰减 $\sim 1/\sqrt{|\mathbf{r}|}$，比热传导的指数衰减慢得多——信号可以传播到更远的位置，这正是 CASSI 重建需要的：从高透射率区域恢复远处低透射率区域的信息。

**与 MST Mask-Guided Mechanism 的区别**：

- MST MM：mask 作为 Attention 的 bias，影响统计相关性权重
- H1-γ：mask 作为物理方程的源项强度，影响波场的激励分布

两者互补，可在并联架构中同时使用。

### 3.6 与前文方案 B 的严格区分

前文方案 B（Duhamel 积分）的 Green 函数：

$$G_\text{wave}(\boldsymbol{\omega},t-\tau)=\frac{e^{-\alpha(t-\tau)/2}\sin(\omega_d(t-\tau))}{\omega_d}$$

H1-γ 的 Green 函数（频域）：

$$G_\text{Helm}(\boldsymbol{\omega},\lambda)=\frac{1}{k^2(\lambda)-|\boldsymbol{\omega}|^2+i\epsilon}$$

**根本区别**：
- 前文 B 是**时域 Green 函数**，描述"从 $\tau$ 时刻的激励，到 $t$ 时刻的响应"，需要积分
- H1-γ 是**频域（稳态）Green 函数**，描述"稳态场的空间分布"，直接频域除法，无需积分
- H1-γ 含有 $k^2(\lambda)$（物理波数），前文 B 没有

### 3.7 与前文方案 D（Klein-Gordon）的关系证明

**命题**：H1-γ 是前文方案 D 的稳态极限。

**证明**：KG 方程（前文）：$\partial_{tt}u+\alpha\partial_tu=v^2\nabla^2u-m^2u$

令稳态 $\partial_t u=0$：$v^2\nabla^2u=m^2u$，即 $\nabla^2u-(m/v)^2u=0$。

令 $m^2=-k^2$（$m^2<0$，对应振荡而非指数衰减）：$\nabla^2u+(k/v)^2u=0$。

令 $v=1$：$\nabla^2u+k^2u=0$，**正是亥姆霍兹方程**。$\blacksquare$

因此 H1-γ 和前文方案 D 不是重复，而是互补：KG 描述传播动力学（时域），亥姆霍兹描述共振结构（稳态）。

### 3.8 实现伪代码

```
输入: x_in [B,C,H,W], mask [B,C,H,W]
# C=28，每个通道对应波段 λ_b，k_phys[C] 预计算

# 1. 源场编码
s = source_encoder(x_in)          # DWConv+GELU+Conv1x1, [B,C,H,W]

# 2. Mask 源项调制
ms = mask * s                     # 空间域乘法, O(N)

# 3. 2D rFFT（逐通道）
ms_fft = rfft2(ms)                # [B,C,H,W//2+1], O(N log N)

# 4. 亥姆霍兹算子分母
k_sq = k_phys.view(1,C,1,1)**2   # [1,C,1,1]
# 空间频率网格
fh = fftfreq(H).view(1,1,-1,1)
fw = rfftfreq(W).view(1,1,1,-1)
omega_sq = (2*pi)**2 * (fh**2 + fw**2)   # [1,1,H,W//2+1]
denom = k_sq - omega_sq + 1j*eps  # eps 可学习 [1,C,1,1]

# 5. 频域除法
f_fft = ms_fft / denom

# 6. iFFT + 实部
f_out = irfft2(f_fft, s=(H,W))   # [B,C,H,W]

# 7. 输出投影
out = LN(f_out) → Linear → SiLU gate → Linear
```

---

## 4. 方案二 H1-β：复折射率亥姆霍兹（吸收体建模）

### 4.1 复折射率光学基础

在有吸收的介质（金属、染料、不透明涂层）中，折射率是复数：

$$\tilde{n}(\mathbf{r},\lambda)=n(\mathbf{r},\lambda)+i\kappa(\mathbf{r},\lambda) \tag{4.1}$$

其中 $n$ 是实折射率（相速度），$\kappa\geq0$ 是消光系数（吸收强度）。

复波数：

$$\tilde{k}(\mathbf{r},\lambda)=\frac{2\pi\tilde{n}}{\lambda}=\underbrace{\frac{2\pi n}{\lambda}}_{k_r}+i\underbrace{\frac{2\pi\kappa}{\lambda}}_{k_i} \tag{4.2}$$

平面波传播：

$$E\propto e^{i\tilde{k}x}=\underbrace{e^{ik_r x}}_{\text{振荡}}\cdot\underbrace{e^{-k_i x}}_{\text{衰减}} \tag{4.3}$$

虚部 $k_i>0$ 产生指数衰减，这是 Beer-Lambert 定律的微观起源。

### 4.2 CASSI Mask 的吸收体类比

CASSI mask 的透射率 $M(x,y)$ 与消光系数的关系（薄层近似，传播距离 $d$）：

$$M(x,y)=e^{-4\pi\kappa(x,y)d/\lambda} \tag{4.4}$$

反解：

$$\kappa(x,y,\lambda)=-\frac{\lambda}{4\pi d}\ln M(x,y) \tag{4.5}$$

在网络中用可学习参数软化：

$$k_i(x,y,\lambda)=\underbrace{\kappa_0}_{\text{可学习}}\cdot(1-M(x,y))\cdot\underbrace{\frac{2\pi}{\lambda_b}}_{\text{物理}} \tag{4.6}$$

$\kappa_0\geq0$ 控制总体吸收强度；$1-M$ 使低透射率区域有高吸收；$2\pi/\lambda_b$ 引入波长依赖（短波吸收强）。

### 4.3 复亥姆霍兹方程展开

把 $\tilde{k}=k_r+ik_i$ 代入亥姆霍兹方程（加源项）：

$$\nabla^2 f+\tilde{k}^2(x,y,\lambda)f=-s \tag{4.7}$$

$$\tilde{k}^2=(k_r+ik_i)^2=\underbrace{(k_r^2-k_i^2)}_{\text{实部}}+i\underbrace{2k_rk_i}_{\text{虚部}} \tag{4.8}$$

$$\nabla^2 f+(k_r^2-k_i^2)f+2ik_rk_i\,f=-s \tag{4.9}$$

**空间依赖的 $k_i(x,y)$ 使方程变系数**，频域方法失效。采用**Strang 算子分裂**（二阶精度）。

### 4.4 Strang 分裂推导

把方程 (4.9) 拆为两个算子：

$$\mathcal{A}f=\nabla^2f+k_r^2f+i\langle k_i\rangle_\text{avg}(\cdots)\quad\text{（均匀传播，频域可解）}$$

$$\mathcal{B}f=-k_i^2f+2ik_r k_i f-i\langle k_i\rangle_\text{avg}(\cdots)\quad\text{（空间吸收，逐点乘法）}$$

精确的 Strang 分裂：

$$f_\text{out}\approx e^{\mathcal{B}/2}\circ e^{\mathcal{A}}\circ e^{\mathcal{B}/2}[f_\text{in}]$$

**工程简化版**（一阶 Lie 分裂，足够精确）：

**步骤 A**（频域，均匀 $k_r$）：

$$\hat{f}^{(A)}(\boldsymbol{\omega},\lambda)=\frac{\hat{s}(\boldsymbol{\omega},\lambda)}{k_r^2(\lambda)-|\boldsymbol{\omega}|^2+i\epsilon}$$

$$f^{(A)}=\mathcal{F}_{xy}^{-1}[\hat{f}^{(A)}] \tag{4.10}$$

**步骤 B**（空间域，局部吸收）：

$$f_\text{out}(x,y,\lambda)=f^{(A)}(x,y,\lambda)\cdot e^{-k_i(x,y,\lambda)\cdot L} \tag{4.11}$$

$$=f^{(A)}\cdot\exp\!\left(-\kappa_0(1-M)\cdot\frac{2\pi L}{\lambda_b}\right) \tag{4.12}$$

其中 $L$ 是等效传播路径长度（可学习标量）。

### 4.5 Strang 分裂（二阶版）

对精度有要求时用：

$$f^{(B/2)}=s\cdot e^{-\kappa_0(1-M)\pi L/\lambda_b}$$

$$\hat{f}^{(A)}=\frac{\mathcal{F}[f^{(B/2)}]}{k_r^2-|\boldsymbol{\omega}|^2+i\epsilon}$$

$$f^{(A)}=\mathcal{F}^{-1}[\hat{f}^{(A)}]$$

$$f_\text{out}=f^{(A)}\cdot e^{-\kappa_0(1-M)\pi L/\lambda_b} \tag{4.13}$$

**误差阶**：Lie 分裂 $O(\delta)$，Strang 分裂 $O(\delta^2)$，其中 $\delta=\kappa_0 L/\lambda_b$ 是吸收强度参数。当 $\kappa_0$ 较小（弱吸收假设）时，一阶分裂已足够精确。

### 4.6 波长依赖吸收的物理解读

吸收因子 $e^{-\kappa_0(1-M)\cdot 2\pi L/\lambda_b}$：

- 蓝光（$\lambda_b=400$ nm，$2\pi/\lambda_b$ 大）：吸收强，在 mask=0 处几乎无残留
- 红光（$\lambda_b=700$ nm，$2\pi/\lambda_b$ 小）：吸收弱，mask=0 处保留更多信号

这与真实材料的光学特性一致：大多数不透明涂层对高频光（蓝光）的吸收强于低频光（红光）。

**对 HSI 重建的意义**：蓝光波段在低透射率区域几乎没有信息，重建时需要完全依赖传播（Green 函数）；红光波段在低透射率区域仍有弱信号，可以直接利用。这种**波段差异化处理**是 H1-β 相对 H1-γ 的独特贡献。

### 4.7 共振峰的虚部修正

H1-γ 的分母虚部：$i\epsilon$（正则化）

H1-β 的分母虚部（步骤 A）：$i\epsilon + i\langle 2k_r k_i\rangle$（$\langle\cdot\rangle$ 是空间平均）

H1-β 的有效正则化更强（因为加入了来自吸收的虚部贡献），共振峰天然被展宽——物理上，吸收介质中的共振总是比无吸收介质的共振更宽（Q 因子降低）。

### 4.8 与 H1-γ 的对比

$$\text{H1-β} \xrightarrow{\kappa_0\to 0} \text{H1-γ}$$

H1-β 比 H1-γ 多两个物理效应：
1. **幅值衰减**（步骤 B）：mask 直接控制输出幅度，空间依赖
2. **共振展宽**（步骤 A 的虚部）：吸收展宽共振峰，改变频率响应

两者合起来构成完整的光学吸收体模型。

---

## 5. 方案三 H2-α精化：物理波数注入 WPO 色散关系

### 5.1 核心修改

前文 3D WPO 色散关系：

$$\omega_0^2=v_s^2(\omega_x^2+\omega_y^2)+v_\lambda^2\omega_\lambda^2 \tag{5.1}$$

$v_\lambda$ 是可学习的，$\omega_\lambda$ 是光谱方向的傅里叶频率（需要对光谱维度做 FFT）。

H2-α 的修改：**不对光谱维做 FFT**，而是把物理波数 $k(\lambda)$ 直接作为每个波段的固有频率偏置：

$$\boxed{\omega_0^2(\boldsymbol{\omega},\lambda)=v_s^2(\omega_x^2+\omega_y^2)+k_\text{eff}^2(\lambda)} \tag{5.2}$$

其中：

$$k_\text{eff}(\lambda)=(1-\gamma)\tilde{k}_\text{phys}(\lambda)+\gamma\,k_\text{learn}(\lambda) \tag{5.3}$$

$\gamma\in(0,1)$ 是软硬先验混合比例，$\tilde{k}_\text{phys}$ 是归一化物理波数（预计算，固定），$k_\text{learn}(\lambda)$ 是可学习的修正项（初始化为 $\tilde{k}_\text{phys}$）。

### 5.2 物理解释：波段作为独立谐振子

原始 3D WPO 把 $\lambda$ 当作第三个传播方向（类似 $x,y$）。H2-α 的新解读：

**每个波段 $\lambda$ 是一个独立的谐振子**，其固有频率由物理波长决定：

$$\frac{d^2\hat{f}_\lambda}{dt^2}+\alpha\frac{d\hat{f}_\lambda}{dt}+\left[v_s^2|\boldsymbol{\omega}|^2+k^2(\lambda)\right]\hat{f}_\lambda=0 \tag{5.4}$$

对比前文方案 D（Klein-Gordon）：$m^2\leftrightarrow k^2(\lambda)$。

**区别**：前文方案 D 的 $m^2$ 是可学习参数（从 mask 预测），H2-α 的 $k^2(\lambda)$ 是物理常数（从波长计算）。H2-α 更严格，是方案 D 的物理约束版本。

### 5.3 完整推导

**方程**（每个波段独立，不对 $\lambda$ 做 FFT）：

$$\frac{\partial^2 f_\lambda}{\partial t^2}+\alpha\frac{\partial f_\lambda}{\partial t}=v_s^2\nabla_{xy}^2 f_\lambda-k^2(\lambda)f_\lambda \tag{5.5}$$

对 $(x,y)$ 做 2D 空间 FFT：

$$\frac{d^2\hat{f}_\lambda}{dt^2}+\alpha\frac{d\hat{f}_\lambda}{dt}+\omega_0^2(\boldsymbol{\omega},\lambda)\hat{f}_\lambda=0 \tag{5.6}$$

其中 $\omega_0^2=v_s^2|\boldsymbol{\omega}|^2+k^2(\lambda)$（式 5.2）。

**判别式**：

$$\eta(\boldsymbol{\omega},\lambda)=\omega_0^2-\left(\frac{\alpha}{2}\right)^2=v_s^2|\boldsymbol{\omega}|^2+k^2(\lambda)-\frac{\alpha^2}{4} \tag{5.7}$$

由于 $k^2(\lambda)>0$ 恒成立，相比原始 WPO（$\eta_\text{orig}=v_s^2|\boldsymbol{\omega}|^2-\alpha^2/4$），H2-α 的 $\eta$ 整体**向正方向偏移 $k^2(\lambda)$**。

**物理影响**：物理波数把更多的频率模式推入欠阻尼振荡区（正 $\eta$），使低空间频率模式也能振荡传播而不是纯衰减。尤其对低频背景区域，原始 WPO 可能过阻尼（丢失信息），H2-α 因 $k^2(\lambda)>0$ 可能欠阻尼（保留信息）。

**闭式解**（同前文统一形式，$\eta$ 定义更新）：

$$\hat{f}_\lambda(\boldsymbol{\omega},t)=e^{-\alpha t/2}\!\left[\hat{f}_{0,\lambda}^M\cdot\operatorname{Cs}(\eta,t)+\!\left(\hat{g}_{0,\lambda}^M+\frac{\alpha}{2}\hat{f}_{0,\lambda}^M\right)\operatorname{Sn}(\eta,t)\right] \tag{5.8}$$

$$\operatorname{Cs}(\eta,t)=\begin{cases}\cos(\sqrt{\eta}t)&\eta>0\\\cosh(\sqrt{-\eta}t)&\eta<0\end{cases},\quad \operatorname{Sn}(\eta,t)=\begin{cases}\sin(\sqrt{\eta}t)/\sqrt{\eta}&\eta>0\\\sinh(\sqrt{-\eta}t)/\sqrt{-\eta}&\eta<0\end{cases}$$

其中 $\hat{f}_{0,\lambda}^M=\mathcal{F}_{xy}[M\cdot u_{0,\lambda}]$（方案 A 初始门控，前文已证），$\hat{g}_{0,\lambda}^M=\mathcal{F}_{xy}[M\cdot v_{0,\lambda}]$（速度场门控）。

### 5.4 软硬先验混合的梯度分析

$$k_\text{eff}(\lambda)=(1-\gamma)\tilde{k}_\text{phys}(\lambda)+\gamma k_\text{learn}(\lambda)$$

对 $\gamma$ 的梯度：

$$\frac{\partial k_\text{eff}}{\partial\gamma}=k_\text{learn}-\tilde{k}_\text{phys}$$

训练初期 $k_\text{learn}\approx\tilde{k}_\text{phys}$（初始化相同），梯度接近零，$\gamma$ 几乎不更新——物理先验自然主导。随着 $k_\text{learn}$ 偏离物理值，$\gamma$ 开始有效，允许网络在物理值附近学习修正。

这是一种**隐式课程学习**（curriculum learning）：早期遵从物理，后期允许偏离。

### 5.5 与原始 3D WPO 的计算量对比

| 操作 | 原始 3D WPO | H2-α |
|-----|-----------|------|
| FFT | 3D rFFT（$H\times W\times C$） | 2D rFFT×C（逐波段） |
| 频率维度 | $(\omega_x,\omega_y,\omega_\lambda)$ | $(\omega_x,\omega_y)$ per $\lambda$ |
| $\omega_0^2$ | $v_s^2(|\boldsymbol{\omega}_{xy}|^2)+v_\lambda^2\omega_\lambda^2$ | $v_s^2|\boldsymbol{\omega}_{xy}|^2+k^2(\lambda)$ |
| 内存峰值 | $B\times H\times(W/2+1)\times$ complex | 相同 |
| 参数 | $v_s,v_\lambda,\alpha,t$（4个） | $v_s,\alpha,t,\gamma$（4个）+$k_\text{learn}$（$B$个） |

两者复杂度均 $O(N\log N)$，H2-α 少了对光谱维的 FFT（但光谱维 $B=28$ 小，实际差异不大）。

---

## 6. 方案四 H2-γ：三合一统一框架

### 6.1 设计原理

H2-γ 把前三个方案的核心物理机制合并为一个统一前向传播流程：

| 步骤 | 来源 | 物理角色 |
|-----|------|---------|
| 步骤 1：初始软门控 | 前文方案 A | CASSI 初始编码（mask 控制激励强度） |
| 步骤 2：物理波数 WPO | 方案三 H2-α | 动态传播（时域波动，$k(\lambda)$ 决定振荡频率） |
| 步骤 3：Beer-Lambert 吸收 | 方案二 H1-β 的步骤 B | 输出衰减（mask 控制出口透射率） |

三步形成完整的**光学系统类比**：

```
光源激励（Step 1）→ 介质中传播（Step 2）→ 吸收体衰减（Step 3）
mask 控制入射 →  物理色散传播  →  mask 控制透射
```

### 6.2 完整数学框架

**Step 1：初始条件（软门控，继承方案 A）**

$$u_{0,\lambda}^M(x,y)=[\epsilon+(1-\epsilon)M(x,y)]\cdot\Phi_\lambda(x_\text{in}) \tag{6.1}$$

$$v_{0,\lambda}^M(x,y)=[\epsilon+(1-\epsilon)M(x,y)]\cdot\Psi_\lambda(x_\text{in}) \tag{6.2}$$

$\Phi_\lambda, \Psi_\lambda$ 是语义编码器和速度编码器（DWConv 系列），$\epsilon=0.1$。

**Step 2：物理波数 WPO 传播（H2-α）**

$$\omega_0^2(\boldsymbol{\omega},\lambda)=v_s^2(\omega_x^2+\omega_y^2)+k_\text{eff}^2(\lambda) \tag{6.3}$$

$$\hat{u}_\lambda(\boldsymbol{\omega},t)=e^{-\alpha t/2}\!\left[\hat{u}_{0,\lambda}^M\operatorname{Cs}(\eta,t)+\!\left(\hat{v}_{0,\lambda}^M+\frac{\alpha}{2}\hat{u}_{0,\lambda}^M\right)\!\operatorname{Sn}(\eta,t)\right] \tag{6.4}$$

$$u_\lambda=\mathcal{F}_{xy}^{-1}[\hat{u}_\lambda] \tag{6.5}$$

**Step 3：Beer-Lambert 空间吸收（H1-β 的步骤 B）**

$$f_{\text{out},\lambda}(x,y)=u_\lambda(x,y)\cdot\exp\!\left(-\kappa_0(1-M(x,y))\cdot\frac{2\pi L}{\lambda_b}\right) \tag{6.6}$$

**合并后的统一公式**：

$$\boxed{f_{\text{out},\lambda}=e^{-\kappa_0(1-M)\frac{2\pi L}{\lambda_b}}\cdot\mathcal{F}_{xy}^{-1}\!\left\{e^{-\frac{\alpha t}{2}}\!\left[\hat{u}_{0,\lambda}^M\operatorname{Cs}(\eta,t)+\!\left(\hat{v}_{0,\lambda}^M+\frac{\alpha}{2}\hat{u}_{0,\lambda}^M\right)\!\operatorname{Sn}(\eta,t)\right]\right\}} \tag{6.7}$$

其中 $\eta=v_s^2|\boldsymbol{\omega}|^2+k_\text{eff}^2(\lambda)-(\alpha/2)^2$，$\hat{u}_{0,\lambda}^M=\mathcal{F}_{xy}[[\epsilon+(1-\epsilon)M]\Phi_\lambda(x_\text{in})]$。

### 6.3 Mask 的双重作用证明

**定理 6.1**：在 H2-γ 中，mask $M$ 同时作用于信号的"入口"（Step 1）和"出口"（Step 3），两者的作用不可互相替代。

**证明**：

设 $M=0$（完全遮挡区域）。

Step 1：$u_{0,\lambda}^M=\epsilon\cdot\Phi_\lambda(x_\text{in})$（仍有 $\epsilon$ 的残留激励）

Step 2：WPO 传播把邻近 $M=1$ 区域的能量传到此处，令传播后的场为 $u_\lambda(x,y)\neq0$

Step 3：$f_\text{out}=u_\lambda\cdot e^{-\kappa_0\cdot 2\pi L/\lambda_b}$（对此位置施加强吸收）

若只有 Step 1 没有 Step 3：$M=0$ 处经过 WPO 传播后有较强信号，不符合 CASSI 物理（mask=0 的地方理论上没有测量到光）。

若只有 Step 3 没有 Step 1：Step 3 把传播来的信号也强烈衰减，但初始激励 $\Phi_\lambda$ 的强度不受 mask 控制，物理不自洽。

两者同时存在：入口控制"激励来源"，出口控制"输出幅值"，物理自洽。$\blacksquare$

### 6.4 可学习参数汇总

| 参数 | 维度 | 初始值 | 约束 | 物理角色 |
|-----|------|-------|------|---------|
| $\alpha$ | 标量/层 | 0.1 | softplus | WPO 阻尼 |
| $v_s$ | 标量/层 | 1.0 | softplus | 空间波速 |
| $t$ | 标量/层 | 1.0 | softplus | 传播时间 |
| $\gamma$ | 标量 | 0.1 | sigmoid | 软硬先验混合比 |
| $k_\text{learn}(\lambda)$ | $[B]$ | $\tilde{k}_\text{phys}$ | 无约束 | 波数可学习修正 |
| $\kappa_0$ | 标量/层 | 0.5 | softplus | 基础消光系数 |
| $L$ | 标量/层 | 1.0 | softplus | 等效传播距离 |
| $\epsilon$（Step 1） | 标量 | 0.1 | 固定或微调 | 软门控下限 |

**额外参数量**：约 $7\times\text{num\_layers}+B$ 个标量，极轻量。

### 6.5 消融实验设计

| 消融配置 | 去掉的内容 | 验证的假设 |
|---------|----------|----------|
| Full H2-γ | — | 完整方案 |
| 去掉 Step 3 | Beer-Lambert 吸收 | 吸收修正的必要性 |
| 去掉 Step 1 中的 $M$ | 初始门控 | 入口 mask 的必要性 |
| 随机初始化 $k$（不用物理值） | $k_\text{phys}$ | 物理先验的贡献 |
| 固定 $\gamma=0$（纯物理 $k$） | 可学习修正 | 物理值是否足够 |
| H2-α 单独 | Step 3（仅方案三） | 吸收项 vs 传播项的相对贡献 |
| H1-γ 单独 | Step 2 动态传播 | 静态亥姆霍兹 vs 动态 WPO |

---

## 7. 四方案综合比较与推荐

### 7.1 数学属性全景

| 属性 | H1-γ | H1-β | H2-α | H2-γ |
|-----|------|------|------|------|
| 基础方程 | 非齐次亥姆霍兹 | 复折射率亥姆霍兹 | 物理波数 WPO | 三合一 |
| 时间演化（动态） | ✗ 稳态 | ✗ 稳态 | ✓ | ✓ |
| 物理波数 $k(\lambda)$ | ✓ | ✓ | ✓ | ✓ |
| 空间吸收（Beer-Lambert）| ✗ | ✓ | ✗ | ✓ |
| 共振奇点 | ✓ | ✓ | ✗（无共振结构）| ✗ |
| Mask 入口门控 | 源项 | 源项 | 初始条件 | 初始条件 |
| Mask 出口修正 | ✗ | ✓ | ✗ | ✓ |
| 闭式解 | 完整 | 近似（Strang 分裂）| 完整 | 近似（分裂）|
| 复杂度 | $O(N\log N)$ | $O(N\log N)$ | $O(N\log N)$ | $O(N\log N)$ |
| 前文重叠度 | 低 | 极低（全新）| 中（含新内容）| 低 |
| 实现难度 | ★★ | ★★★ | ★★ | ★★★★ |
| 创新强度 | ★★★ | ★★★★ | ★★★ | ★★★★★ |

### 7.2 推荐优先级

**论文主力方案：H2-γ（★★★★★）**

理由：
- 在一个统一框架中包含三个独立创新点，每个都可以单独作为贡献
- 全套消融实验天然构成（6 种消融，见 §6.5）
- 物理故事完整：激励→传播→吸收，对应 CASSI 光路的全过程
- 与前文方案 A/D 均不重复（组合是新的，物理波数是新的，吸收项是新的）

**作为独立消融的对照：H1-γ（★★★）**

理由：
- 实现最简单（一次 FFT + 频域除法）
- 纯稳态方案，和 H2-γ 对比可量化"动态传播 vs 稳态共振"的贡献差
- 亥姆霍兹算子结构（$1/(k^2-|\boldsymbol{\omega}|^2)$）本身是新颖的，可以单独写一节

**高风险高回报：H1-β（★★★★）**

理由：
- 复折射率建模是前文完全没有的，创新强
- 波长依赖吸收（$2\pi/\lambda_b$）是非常优美的物理先验
- 可以单独作为一种新的 mask 机制提出

**最终推荐的论文实验组合**：

```
Baseline:       MST（前文，作为起点）
+ WPO（前文）:  WaveMST_3D（3D WPO + 方案A）
+ H2-α:        WaveMST_3D 但用物理波数替代 v_λω_λ
+ H1-γ:        纯亥姆霍兹（稳态），验证稳态 vs 动态
+ H2-γ（完整）: 主推方案，预期最优
  ├── -Step3:  去掉吸收项
  ├── -k_phys: 随机初始化 k
  └── -Step1M: 去掉初始 mask 门控
```

### 7.3 论文 Contributions 建议

1. **(C1) 首次将物理光波波数 $k(\lambda)=2\pi/\lambda$ 作为硬先验注入 HSI 重建网络的色散关系**，实现波长-频率的物理对齐

2. **(C2) 提出基于亥姆霍兹方程的频域逆算子（H1-γ/H1-β）**，建立稳态光谱共振与 CASSI mask 的物理联系

3. **(C3) 提出 Beer-Lambert 吸收修正项（H1-β, H2-γ Step 3）**，将 CASSI mask 的光学吸收特性直接编码为波长依赖的指数衰减

4. **(C4) 提出 H2-γ 三合一框架**，统一了初始编码、物理传播、光学吸收三个物理过程

