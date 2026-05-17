# WaveMST 方案深度推导：路线 2/3 × Mask 四方案的数学证明与工程评估

> **作者备注**：本文围绕"WPO + MST + Mask"的方案设计做严格的数学推导和物理论证。重点分析路线 2（空间WPO+光谱WPO）、路线 3（3D WPO）、四种 Mask 添加方案（A/B/C/D）、三种 Transformer 处理方式（舍弃/替代/并联）的利弊。所有推导从基础方程开始，闭式解推导完整。

---

## 目录

1. [基础框架：坐标系与约定](#1-基础框架坐标系与约定)
2. [核心闭式解推导（路线 3 基础）](#2-核心闭式解推导路线-3-基础)
3. [路线 2 vs 路线 3：数学不等价性证明](#3-路线-2-vs-路线-3数学不等价性证明)
4. [四种 Mask 方案数学分析](#4-四种-mask-方案数学分析)
5. [Transformer 处理的三条路线](#5-transformer-处理的三条路线)
---

## 1. 基础框架：坐标系与约定

### 1.1 变量定义

HSI 特征张量 $u(x, y, \lambda, t)$，各维度含义：
- $(x, y) \in [0, H] \times [0, W]$：空间坐标
- $\lambda \in \{1, 2, \dots, B\}$：光谱波段（离散，通常 $B = 28$ 或 $100多的都有$）
- $t$：传播时间（对应网络深度，每个 WPO Block 对应一个 $t$ 步长）
- 通道维 $c = 1, \dots, C$ 在下文中隐含（通常与 $\lambda$ 混用，但严格来说 $C$ 是特征通道数，$B$ 是物理波段数）

CASSI 物理掩模：$M(x, y) \in [0, 1]$。**关键事实**：CASSI 的编码孔径是一块物理掩模板，**只在空间上有结构，沿光谱方向无调制**。这个物理约束决定了 mask 在数学建模中的角色边界。

### 1.2 傅里叶变换约定

3D 傅里叶变换：
$$\hat{u}(\boldsymbol{\omega}, t) = \int u(\mathbf{r}, t) e^{-i\boldsymbol{\omega}\cdot\mathbf{r}} d\mathbf{r}, \quad \boldsymbol{\omega} = (\omega_x, \omega_y, \omega_\lambda)$$

频域微分：$\mathcal{F}[\partial_x^2 u] = -\omega_x^2 \hat{u}$，$\mathcal{F}[\partial_t^n u] = \partial_t^n \hat{u}$。

空间 2D 变换单独记为 $\hat{u}_{xy}$，光谱 1D 变换记为 $\hat{u}_\lambda$。

---

## 2. 核心闭式解推导（路线 3 基础）

### 2.1 各向异性 3D 阻尼波方程

路线 3 的基本方程：
$$\frac{\partial^2 u}{\partial t^2} + \alpha \frac{\partial u}{\partial t} = v_s^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + v_\lambda^2 \frac{\partial^2 u}{\partial \lambda^2} \tag{2.1}$$

初始条件：
- $u(\mathbf{r}, 0) = u_0(\mathbf{r})$（语义场）
- $\partial_t u|_{t=0} = v_0(\mathbf{r})$（语义速度场）

各向异性体现在 $v_s \neq v_\lambda$——空间波速和光谱波速独立，因为 HSI 中空间和光谱的物理尺度差异巨大（有待商榷）（256 像素 vs 28 波段）。

### 2.2 频域 ODE 导出

对 (2.1) 两边做 3D 傅里叶变换：

$$\frac{d^2 \hat{u}}{dt^2} + \alpha \frac{d\hat{u}}{dt} + \omega_0^2(\boldsymbol{\omega}) \hat{u} = 0 \tag{2.2}$$

其中**固有频率**（各向异性）：
$$\boxed{\omega_0^2(\boldsymbol{\omega}) = v_s^2(\omega_x^2 + \omega_y^2) + v_\lambda^2 \omega_\lambda^2} \tag{2.3}$$

这是每个 $\boldsymbol{\omega}$ 点上独立的二阶常系数线性 ODE。

### 2.3 色散关系分析

特征方程 $r^2 + \alpha r + \omega_0^2 = 0$ 的根：
$$r_{\pm} = -\frac{\alpha}{2} \pm \sqrt{(\alpha/2)^2 - \omega_0^2}$$

判别式 $\Delta = (\alpha/2)^2 - \omega_0^2$ 决定三种行为：

**欠阻尼区**（$\omega_0 > \alpha/2$，即 $\Delta < 0$）
$$r_{\pm} = -\frac{\alpha}{2} \pm i\omega_d, \quad \omega_d = \sqrt{\omega_0^2 - (\alpha/2)^2}$$

**过阻尼区**（$\omega_0 < \alpha/2$，即 $\Delta > 0$）
$$r_{\pm} = -\frac{\alpha}{2} \pm \gamma, \quad \gamma = \sqrt{(\alpha/2)^2 - \omega_0^2}$$

**临界阻尼**（$\omega_0 = \alpha/2$）：重根 $r = -\alpha/2$，解形如 $(A+Bt)e^{-\alpha t/2}$。数值上是零测度集，可忽略。

**物理区划示意**：

```
频域 ω-空间 被分为两个区域:

    |ω|
     │   [振荡衰减区]
     │   高频: 快速振荡 + 缓慢衰减
     │   ── 保留边缘/纹理
 α/2 ├─────────────────
     │   [纯衰减区]  
     │   低频: 无振荡, 纯指数衰减
     │   ── 逐步抑制慢变背景
     └──────────────────→ t
```

**HSI 任务含义**：
- 高频区（边缘、地物轮廓）处于振荡区 → 信息以振荡形式保留
- 低频区（均匀背景）处于纯衰减区 → 信息逐步被阻尼滤除

这恰好匹配 HSI 任务的需求（我们想保留地物轮廓，衰减背景冗余）。

### 2.4 欠阻尼情况闭式解

通解 $\hat{u} = e^{-\alpha t/2}[A\cos(\omega_d t) + B\sin(\omega_d t)]$。

应用初始条件：
- $\hat{u}(\boldsymbol{\omega}, 0) = \hat{u}_0 \Rightarrow A = \hat{u}_0$
- $\partial_t \hat{u}\big|_{t=0} = \hat{v}_0$

第二个条件需要计算导数：
$$\partial_t \hat{u}\big|_{t=0} = -\frac{\alpha}{2}\hat{u}_0 + \omega_d B = \hat{v}_0$$

解得 $B = \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\omega_d}$。

**欠阻尼闭式解**：
$$\hat{u}(\boldsymbol{\omega}, t) = e^{-\alpha t/2}\left[\hat{u}_0 \cos(\omega_d t) + \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\omega_d}\sin(\omega_d t)\right] \tag{2.4}$$

### 2.5 过阻尼情况闭式解

通解 $\hat{u} = e^{-\alpha t/2}[A\cosh(\gamma t) + B\sinh(\gamma t)]$。

应用初始条件得到：
$$\hat{u}(\boldsymbol{\omega}, t) = e^{-\alpha t/2}\left[\hat{u}_0 \cosh(\gamma t) + \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\gamma}\sinh(\gamma t)\right] \tag{2.5}$$

### 2.6 统一闭式解

定义符号依赖的"广义正弦/余弦"：

$$\operatorname{Cs}(\eta, t) = \begin{cases} \cos(\sqrt{\eta} \cdot t) & \eta > 0 \\ \cosh(\sqrt{-\eta} \cdot t) & \eta < 0 \end{cases}$$

$$\operatorname{Sn}(\eta, t) = \begin{cases} \sin(\sqrt{\eta}\cdot t)/\sqrt{\eta} & \eta > 0 \\ \sinh(\sqrt{-\eta}\cdot t)/\sqrt{-\eta} & \eta < 0 \end{cases}$$

其中 $\eta = \omega_0^2 - (\alpha/2)^2$。统一闭式解：

$$\boxed{\hat{u}(\boldsymbol{\omega}, t) = e^{-\alpha t/2}\left[\hat{u}_0 \cdot \operatorname{Cs}(\eta, t) + \left(\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0\right) \cdot \operatorname{Sn}(\eta, t)\right]} \tag{2.6}$$

**WaveFormer 原论文的缺陷**：论文只写了欠阻尼情况，默认 $\omega_d^2 > 0$。当 $\alpha$ 较大或 $\omega_0$ 较小时会出现 $\omega_d^2 < 0$（即 $\omega_d$ 变为虚数），此时原论文的 `torch.sqrt(omega_d_sq)` 会产生 NaN。严格实现必须处理这一情况。

### 2.7 能量分析

定义频域能量密度：
$$E(\boldsymbol{\omega}, t) = \frac{1}{2}|\partial_t \hat{u}|^2 + \frac{1}{2}\omega_0^2 |\hat{u}|^2$$

对 (2.1) 做能量泛函分析，可得（对 $t$ 求导）：
$$\frac{dE}{dt} = -\alpha \left|\partial_t \hat{u}\right|^2 \leq 0$$

能量单调递减。这保证了数值稳定性——无论 $\alpha, v_s, v_\lambda$ 如何学习，总能量不会爆炸。

**反差**：热传导方程的能量单调递减且每个频率按 $e^{-2k\omega^2 t}$ 衰减（频率相关），波动方程的能量按 $e^{-\alpha t}$ 整体衰减（频率无关），这正是 WaveFormer 相对 vHeat 的核心优势。

---

## 3. 路线 2 vs 路线 3：数学不等价性证明

### 3.1 路线 2 的数学模型

路线 2 是两个独立的波动方程用**算子分裂**（Lie-Trotter splitting）串联：

**步骤 A（空间 WPO）**：对每个波段 $\lambda$ 独立求解
$$\frac{\partial^2 u}{\partial t^2} + \alpha_s \frac{\partial u}{\partial t} = v_s^2(\partial_x^2 + \partial_y^2) u \tag{3.1}$$

传播 $t_s$ 时间后得到中间态 $\tilde{u}$。

**步骤 B（光谱 WPO）**：对每个空间位置 $(x,y)$ 独立求解
$$\frac{\partial^2 u}{\partial t^2} + \alpha_\lambda \frac{\partial u}{\partial t} = v_\lambda^2 \partial_\lambda^2 u \tag{3.2}$$

以 $\tilde{u}$ 为初始条件，再传播 $t_\lambda$ 时间得到最终输出。

### 3.2 频域算子形式

定义**传播算子** $\mathcal{P}(\boldsymbol{\omega}, t; \alpha, v)$（忽略速度场初始条件，简化分析）：
$$\mathcal{P}_3(\boldsymbol{\omega}, t) = e^{-\alpha t/2}\cos\left(\sqrt{v_s^2(\omega_x^2+\omega_y^2) + v_\lambda^2\omega_\lambda^2 - (\alpha/2)^2} \cdot t\right)$$

路线 3 直接用 $\mathcal{P}_3$。

路线 2 的算子组合为：
$$\mathcal{P}_2(\boldsymbol{\omega}, t_s, t_\lambda) = \underbrace{e^{-\alpha_\lambda t_\lambda/2}\cos\left(\sqrt{v_\lambda^2\omega_\lambda^2 - (\alpha_\lambda/2)^2} \cdot t_\lambda\right)}_{\text{光谱传播算子}} \cdot \underbrace{e^{-\alpha_s t_s/2}\cos\left(\sqrt{v_s^2(\omega_x^2+\omega_y^2) - (\alpha_s/2)^2}\cdot t_s\right)}_{\text{空间传播算子}}$$

### 3.3 不等价性的严格证明

**定理 3.1**：当 $v_s > 0$ 且 $v_\lambda > 0$ 同时非零时，对任意 $t_s, t_\lambda > 0$，不存在 $t_3$ 使得 $\mathcal{P}_2 \equiv \mathcal{P}_3$ 在所有 $\boldsymbol{\omega}$ 上成立。

**证明**：
简化为无阻尼情况（$\alpha=0$）。路线 3 的"相位函数"：
$$\phi_3(\boldsymbol{\omega}, t_3) = \sqrt{v_s^2(\omega_x^2+\omega_y^2) + v_\lambda^2\omega_\lambda^2} \cdot t_3$$

路线 2 的"相位函数"（通过积化和差）：
$$\cos(\phi_2^{(s)})\cos(\phi_2^{(\lambda)}) = \frac{1}{2}\left[\cos(\phi_2^{(s)} + \phi_2^{(\lambda)}) + \cos(\phi_2^{(s)} - \phi_2^{(\lambda)})\right]$$

其中 $\phi_2^{(s)} = v_s\sqrt{\omega_x^2+\omega_y^2}\cdot t_s$，$\phi_2^{(\lambda)} = v_\lambda|\omega_\lambda| t_\lambda$。

路线 3 是单个余弦函数，路线 2 是**两个余弦的叠加**（频率分别是空间+光谱频率之和与之差）。代数上二者不可能相等（除非某个速度为零）。

更直观的角度：路线 3 的等相面是**椭球**
$$v_s^2 t_3^2(\omega_x^2 + \omega_y^2) + v_\lambda^2 t_3^2 \omega_\lambda^2 = \text{const}$$

而路线 2 的等相面是**双曲线/双叶面**。几何不同 → 数学不等价。$\blacksquare$

### 3.4 物理含义区别

| 性质 | 路线 2 | 路线 3 |
|------|-------|-------|
| 波传播方向 | 先全空间传播，再全光谱传播 | 在空-谱超空间中统一传播 |
| 空谱耦合 | 解耦（一个接一个） | 耦合（同时演化） |
| 物理直观 | 像 CT 扫描：先空间成像再光谱展开 | 像 3D 声场：空谱扰动一起振荡 |
| 信息跨模态交互 | 仅在块交界处发生 | 每个瞬时都交互 |
| 参数量 | $2\times$（两套 $\alpha,v$） | $1\times$（一套 $\alpha$，两个 $v$） |
| 复杂度 | $O(N\log N)$（两次 FFT） | $O(N\log N)$（一次 3D FFT） |
| 计算效率 | 略高（2D FFT 比 3D FFT 快） | 略低但差距不大 |

### 3.5 HSI 任务适配度分析

**支持路线 3 的理由**：
1. HSI 的核心特性"same material, similar spectra"是**空谱本质耦合**——空间相邻像素倾向于同时在多个波段相关，3D 波动自然建模这种耦合
2. CASSI 的物理测量也是空谱耦合的（每个空间位置的光谱被 mask 编码后叠加到 2D 传感器上）
3. 模型与物理更同构

**支持路线 2 的理由**：
1. 空谱尺度高度不对称（256×256 vs 28）使 3D FFT 可能有数值问题
2. 两个 WPO 的阻尼/波速分别学习，表达能力更灵活
3. 消融实验更容易：可分别关闭空间或光谱 WPO 看贡献
4. 工程实现更简单（PyTorch 2D FFT 原生支持好，光谱维度用矩阵乘代替 1D FFT）

**最终判决**：两条路线都有论文价值。
- **主论文**建议用路线 3（更自然的物理建模，更有故事）
- **路线 2** 作为消融对照组（"我们发现耦合比解耦好，提升了 X%"）

---

## 4. 四种 Mask 方案数学分析

### 4.1 方案 A：Mask 作为初始振幅门控（★ 推荐）

**数学形式**：

硬门控：
$$u_0^{\text{masked}}(x,y,\lambda) = M(x,y) \cdot u_0(x,y,\lambda) \tag{4.1}$$

软门控（你的工程修正）：
$$u_0^{\text{soft}}(x,y,\lambda) = [\epsilon + (1-\epsilon)M(x,y)] \cdot u_0(x,y,\lambda), \quad \epsilon \in [0, 0.2] \tag{4.2}$$

**闭式解推导（完整）**

将 (4.1) 代入闭式解 (2.6)。由于傅里叶变换的卷积定理：
$$\hat{u}_0^{\text{masked}}(\boldsymbol{\omega}) = \mathcal{F}[M \cdot u_0] = \hat{M}(\omega_x, \omega_y) *_{xy} \hat{u}_0(\boldsymbol{\omega}) \tag{4.3}$$

其中 $*_{xy}$ 是仅在空间频率上的 2D 卷积，光谱频率 $\omega_\lambda$ 维度上不卷积（因为 $M$ 沿 $\lambda$ 是常数）。

**代入闭式解**：
$$\hat{u}(\boldsymbol{\omega}, t) = e^{-\alpha t/2}\left[(\hat{M} *_{xy} \hat{u}_0) \operatorname{Cs}(\eta, t) + (\hat{v}_0 + \frac{\alpha}{2}\hat{M} *_{xy} \hat{u}_0) \operatorname{Sn}(\eta, t)\right] \tag{4.4}$$

**$O(N\log N)$ 保持性**：

读者可能担心"卷积岂不是 $O(N^2)$？" 实际上实现时**不需要在频域做卷积**——直接在空间域做乘法再 FFT 即可：

```
u0_masked = M * u0           # O(N) 乘法
fft_u0 = FFT(u0_masked)      # O(N log N)
apply closed-form modulation  # O(N)
output = IFFT(...)           # O(N log N)
```

总复杂度仍为 $O(N\log N)$。

**物理意义深度分析**

CASSI 的物理过程：
1. 场景光 $I(x,y,\lambda)$ 通过掩模 $M(x,y)$：$I' = M \cdot I$
2. 分光元件做色散偏移：$I''(x,y,\lambda) = I'(x - d(\lambda), y, \lambda)$
3. 传感器积分：$g(x,y) = \int I''(x,y,\lambda) d\lambda$

方案 A 在模型中用 $M$ 乘初始场 $u_0$，**完全对应 CASSI 的步骤 1**——这是最直接的物理建模。

**传播后能量分析**

由 Parseval 定理，初始能量：
$$\|u_0^{\text{masked}}\|_2^2 = \sum_{xyt\lambda} M^2 u_0^2 \leq \|u_0\|_2^2$$

能量衰减比例 $\eta_M = \|M u_0\|^2 / \|u_0\|^2$。CASSI 中典型 $M$ 的透射率约 50%，所以 $\eta_M \approx 0.5$。

传播后能量：$\|u(t)\|^2 \leq e^{-\alpha t} \|u_0^{\text{masked}}\|^2$（来自 2.7 节的能量不等式）。

**理论保证**：方案 A 保持了 (2.6) 闭式解结构，只是初始条件变化。所有数值稳定性分析沿用 WaveFormer 原证明。

**工程注意事项**：
1. $M$ 本身是二值（或 0-1 连续）的，但网络学习会想让它连续。可以让 $M$ 经过一个 $\sigma$ 门控 + bias 项变成软 mask
2. 对不同层 WPO 是否用同一个 $M$？选择有两种：
   - **静态 mask**（所有 WPO 层共享）：更严格的物理约束
   - **动态 mask**（每层从特征预测）：更灵活，但偏离物理
3. 速度场初始化 $v_0$ 的处理：同样乘 $M$ 或者用专门的速度编码器 $\Psi$ 再乘

### 4.2 方案 B：Mask 作为源项（非齐次波方程）

**数学形式**：
$$\frac{\partial^2 u}{\partial t^2} + \alpha\frac{\partial u}{\partial t} = v_s^2\nabla_{xy}^2 u + v_\lambda^2\partial_\lambda^2 u + M(x,y) \cdot S(x,y,\lambda,t) \tag{4.5}$$

其中 $S$ 是源强度（可学习）。

**闭式解推导（Duhamel 原理）**

线性 ODE 的非齐次解 = 齐次解 + 特解。齐次解就是 (2.6)。特解通过 Duhamel 积分：

对 (4.5) 做傅里叶变换：
$$\partial_t^2 \hat{u} + \alpha \partial_t \hat{u} + \omega_0^2 \hat{u} = \widehat{MS}(\boldsymbol{\omega}, t) \tag{4.6}$$

这是强迫振荡方程。齐次方程的 Green 函数（欠阻尼情况）：
$$G(\boldsymbol{\omega}, t) = \frac{e^{-\alpha t/2}\sin(\omega_d t)}{\omega_d} \Theta(t)$$

$\Theta$ 是阶跃函数。通过 Laplace 变换或直接验证可证明 $G$ 满足 $\partial_t^2 G + \alpha\partial_t G + \omega_0^2 G = \delta(t)$。

**特解**：
$$\hat{u}_p(\boldsymbol{\omega}, t) = \int_0^t G(\boldsymbol{\omega}, t-\tau) \widehat{MS}(\boldsymbol{\omega}, \tau) d\tau \tag{4.7}$$

**时间无关源的简化**

若 $S$ 不依赖时间（$S(\mathbf{r},\tau) = S(\mathbf{r})$），则：
$$\hat{u}_p(\boldsymbol{\omega}, t) = \widehat{MS}(\boldsymbol{\omega}) \int_0^t \frac{e^{-\alpha(t-\tau)/2}\sin(\omega_d(t-\tau))}{\omega_d} d\tau \tag{4.8}$$

令 $s = t - \tau$，$ds = -d\tau$，积分区间 $[0, t]$：

$$I(t) = \frac{1}{\omega_d}\int_0^t e^{-\alpha s/2}\sin(\omega_d s) ds \tag{4.9}$$

这是标准积分。用复指数法：$\sin(\omega_d s) = \Im[e^{i\omega_d s}]$

$$\int_0^t e^{-\alpha s/2 + i\omega_d s} ds = \frac{e^{(-\alpha/2 + i\omega_d)t} - 1}{-\alpha/2 + i\omega_d}$$

取虚部：
$$I(t) = \frac{1}{\omega_0^2}\left[1 - e^{-\alpha t/2}\left(\cos(\omega_d t) + \frac{\alpha}{2\omega_d}\sin(\omega_d t)\right)\right] \tag{4.10}$$

其中用到 $(-\alpha/2)^2 + \omega_d^2 = \omega_0^2$。

**完整闭式解**：
$$\boxed{\hat{u}(\boldsymbol{\omega}, t) = \underbrace{e^{-\alpha t/2}\left[\hat{u}_0\cos(\omega_d t) + \frac{\hat{v}_0 + \frac{\alpha}{2}\hat{u}_0}{\omega_d}\sin(\omega_d t)\right]}_{\text{齐次部分(初始条件)}} + \underbrace{\widehat{MS}(\boldsymbol{\omega}) \cdot I(t)}_{\text{源项贡献}}} \tag{4.11}$$

**物理意义分析**

- Mask 在 $M\cdot S$ 中作为**空间调制**，源项 $S$ 作为**光谱+幅度信号注入**
- 长时极限：$\lim_{t\to\infty} I(t) = 1/\omega_0^2$，特解收敛到 $\widehat{MS}/\omega_0^2$——静态响应（稳态）
- $I(t)$ 的物理含义：每个频率 $\omega_0$ 对源项的**共振响应**

**问题与局限**

1. **$S$ 的来源不明**：
   - 如果 $S = u_0$，则方案 B 退化为方案 A 的修改版（初始条件+源项都用 $M u_0$）
   - 如果 $S$ 是可学习参数，失去物理意义
   - 如果 $S$ 来自另一个分支（例如浅层特征），引入额外网络开销
2. **CASSI 物理不完全匹配**：CASSI 的 mask 是**一次性调制**（分光前的一次乘法），不是持续的源项输入
3. **多步传播的复杂性**：如果 $t$ 在每个 Block 都推进一步，那源项 $S$ 到底是每步都注入还是只在 $t=0$？这需要明确定义

**工程实现**

$I(t)$ 是标量函数（可预计算），总复杂度仍 $O(N\log N)$。但源项 $\widehat{MS}$ 的计算需要额外 FFT。实际总计算量约是方案 A 的 1.5 倍。

**适用性评估**：数学自洽，但物理动机相对牵强。除非有特殊设计使 $S$ 有明确物理解释，否则不推荐作为主力方案。

### 4.3 方案 C：Mask 作为频域门控

**数学形式**（两个子方案）：

C1：纯频域调制
$$\hat{u}_{\text{modulated}}(\boldsymbol{\omega}, t) = \hat{u}(\boldsymbol{\omega}, t) \cdot H(\omega_x, \omega_y) \tag{4.12}$$

其中 $H$ 是某种**频率依赖的 mask 调制器**：
$$H(\omega_x, \omega_y) = [1 - \beta(\omega_x, \omega_y)] + \beta(\omega_x, \omega_y) \hat{M}(\omega_x, \omega_y)$$

$\beta$ 是频率加权函数：低频处 $\beta \to 0$（mask 不起作用），高频处 $\beta \to 1$（mask 完全作用）。

C2：每层不同 mask
在每个 WPO 层应用不同的频域调制 $H^{(l)}$。

**与方案 A 的等价性分析（重要！）**

考虑单层情况下 C1 的数学含义。令 $u$ 是 WPO 的输出：
$$u_{\text{out}} = \mathcal{F}^{-1}[H \cdot \mathcal{F}[u]]$$

由卷积定理：
$$u_{\text{out}}(\mathbf{r}) = \tilde{H}(\mathbf{r}_{xy}) *_{xy} u(\mathbf{r}) \tag{4.13}$$

其中 $\tilde{H} = \mathcal{F}_{xy}^{-1}[H]$。这是**空间卷积**——相当于用一个"有效 mask"对输出做模糊化。

**定理 4.1（C 与 A 的等价性）**：在线性 WPO 框架下，方案 C 等价于：对初始条件应用"修正后的 mask" $\tilde{H}$，再做普通 WPO 传播。即
$$\text{方案 C} \equiv \text{方案 A with mask } \tilde{H}$$

**证明**：
波传播算子 $\mathcal{P}$ 是空间上的卷积算子（对应频域乘法）。任意两个频域乘法可交换：
$$\mathcal{F}^{-1}[H \cdot \mathcal{P}(\boldsymbol{\omega}) \cdot \hat{u}_0] = \mathcal{F}^{-1}[\mathcal{P}(\boldsymbol{\omega}) \cdot H \cdot \hat{u}_0] = \mathcal{P} * (\tilde{H} * u_0)$$

即"先传播再频域调制"等价于"先用 $\tilde{H}$ 卷积再传播"。$\blacksquare$

**这意味着什么**：
- **单层的方案 C 本质上是方案 A 的一个特殊版本**（用 $\tilde{H}$ 代替 $M$）
- 方案 C 相对方案 A 的**真正差异**只体现在 $\tilde{H}$ 与 $M$ 不同时
- 如果 $\tilde{H}$ 是从 $\hat{M}$ 经过频率加权得到，那 $\tilde{H}$ 本质是 $M$ 的**模糊化/平滑化版本**

**方案 C 的非平凡版本（C2）**

要让方案 C **真正不同于方案 A**，需要：
1. **每层的 $H^{(l)}$ 不同**（即 mask 在深度方向变化）
2. $H$ 依赖于**光谱频率 $\omega_\lambda$**（即 $H = H(\omega_x, \omega_y, \omega_\lambda)$，不仅依赖空间频率）

光谱频率依赖的 $H$ 打破了 $M$ 只在空间上调制的假设——这偏离了 CASSI 的物理模型，但可能带来更强表达能力。

**适用性评估**：
- **单层 C1**：数学上等价于方案 A（带平滑 mask），无实质差异
- **多层 C2**：有非平凡差异，但光谱频率依赖偏离 CASSI 物理
- **推荐**：作为方案 A 的 ablation，不作为独立方案

### 4.4 方案 D：Mask 作为 Klein-Gordon 质量场

**数学形式**：
$$\frac{\partial^2 u}{\partial t^2} + \alpha\frac{\partial u}{\partial t} = v_s^2\nabla_{xy}^2 u + v_\lambda^2 \partial_\lambda^2 u - m^2(x,y) \cdot u \tag{4.14}$$

其中质量场由 mask 调制：
$$m^2(x,y) = m_0^2 \cdot [1 - M(x,y)] \tag{4.15}$$

物理含义：低透射率 $M \approx 0$ 处，$m^2 \approx m_0^2$ 大（高惯性、强抑制）；高透射率 $M \approx 1$ 处，$m^2 \approx 0$（退化为普通波动）。

**数学挑战：空间依赖的质量场破坏闭式解**

$m^2(x,y)$ 若依赖空间坐标，傅里叶变换会变为：
$$\mathcal{F}[m^2(x,y) u(x,y,\lambda,t)] = \hat{m^2}(\omega_x,\omega_y) *_{xy} \hat{u}(\boldsymbol{\omega}, t) \tag{4.16}$$

频域 ODE 变为：
$$\partial_t^2 \hat{u} + \alpha\partial_t\hat{u} + \omega_0^2 \hat{u} + (\hat{m^2} *_{xy} \hat{u}) = 0 \tag{4.17}$$

这不再是每个 $\boldsymbol{\omega}$ 独立的 ODE，而是**所有空间频率点通过卷积耦合**的积分-微分方程。WaveFormer 的 $O(N\log N)$ 频域解法**失效**。

严格求解需要 $O(N^2)$ 复杂度或数值迭代。

**救赎方案 D1：Born 近似（微扰展开）**

假设 $m^2$ 的作用相比波动主项较弱，把 $-m^2 u$ 项作为微扰处理。

**零阶方程**（不含质量项）：
$$\partial_t^2 u^{(0)} + \alpha\partial_t u^{(0)} = v_s^2\nabla_{xy}^2 u^{(0)} + v_\lambda^2\partial_\lambda^2 u^{(0)}$$

零阶解 $u^{(0)}$ 就是 WaveFormer 闭式解 (2.6)。

**一阶修正方程**：
$$\partial_t^2 u^{(1)} + \alpha\partial_t u^{(1)} - v_s^2\nabla_{xy}^2 u^{(1)} - v_\lambda^2\partial_\lambda^2 u^{(1)} = -m^2(x,y) u^{(0)}(x,y,\lambda,t) \tag{4.18}$$

右端是**已知源项**（$u^{(0)}$ 已求出）。这回到了方案 B 的非齐次波方程形式。

用 Duhamel 积分求解：
$$\hat{u}^{(1)}(\boldsymbol{\omega}, t) = -\int_0^t G(\boldsymbol{\omega}, t-\tau) \cdot \mathcal{F}[m^2(x,y) u^{(0)}(\mathbf{r},\tau)] d\tau \tag{4.19}$$

实际计算时不需要真正做卷积，而是**在空间域做乘法**：
1. 计算 $u^{(0)}(\mathbf{r}, \tau)$（每个 $\tau$ 只需要频域调制+IFFT）
2. 空间乘法 $m^2 \cdot u^{(0)}$
3. FFT 回频域
4. 乘 Green 函数 $G$
5. 对 $\tau$ 积分

**复杂度**：每一步 $O(N\log N)$，积分离散化为 $K$ 步，总复杂度 $O(KN\log N)$。$K=5\sim10$ 时实用。

**近似误差**：Born 近似的误差量级为 $O((m_0^2 t / \omega_0^2)^2)$，要求 $m_0^2 t / \omega_0^2 \ll 1$。HSI 中典型 $\omega_0 \sim O(1)$，所以 $m_0^2 t$ 必须是小量。工程上可以限制 $m_0$ 的范围（如 $m_0^2 < 0.1$）。

**救赎方案 D2：退化到常数质量（空间无关）**

若 $m^2$ 退化为标量常数（不依赖空间坐标），则 $m^2 u$ 在频域不产生耦合：
$$\mathcal{F}[m^2 u] = m^2 \hat{u}$$

色散关系修正为：
$$\omega_0^2(\boldsymbol{\omega}) = v_s^2(\omega_x^2+\omega_y^2) + v_\lambda^2\omega_\lambda^2 + m^2 \tag{4.20}$$

闭式解完全同 WaveFormer。

**物理意义**：$+m^2$ 引入**截止频率** $\omega_{\min} = m$。所有 $\omega_0 < m$ 的模都进入过阻尼区（纯衰减无振荡）。

**但 mask 信息去哪了？** 若 $m^2$ 是常数，mask 没有被利用。这就退化为普通 Klein-Gordon，与 mask 无关。

**折中方案 D\***：mask 用于全局调制

$$m^2_{\text{eff}} = m_0^2 \cdot \langle 1 - M \rangle_\Omega \tag{4.21}$$

其中 $\langle\cdot\rangle_\Omega$ 是空间平均。即用 mask 的**全局透射率**（一个标量）来调节质量。这保留了闭式解，同时利用了 mask 的整体信息（但丢失了空间细节）。

**救赎方案 D3：仅光谱依赖的质量**

若 $m^2 = m^2(\lambda)$ 仅依赖光谱：
$$\mathcal{F}_{xy}[m^2(\lambda) u] = m^2(\lambda) \hat{u}_{xy}$$

空间傅里叶变换下 $m^2$ 是常数。但对光谱方向做 FFT 后：
$$\mathcal{F}_\lambda[m^2(\lambda) u] = \hat{m^2}(\omega_\lambda) *_\lambda \hat{u}_\lambda$$

仍然需要在光谱频率方向做卷积。$O(B^2)$ 当 $B=28$ 时仅 $784$ 次操作，可接受。

但 mask 只依赖空间，不依赖光谱——所以 D3 的 $m^2(\lambda)$ 无法用 mask 定义。要用 D3 必须抛弃 mask 对应关系。

### 4.5 方案 D 的最终分析

| 变种 | 闭式解 | 复杂度 | 与 mask 的关系 | 物理合理性 |
|------|-------|--------|-------------|-----------|
| D 原始（$m^2(x,y)$ 依赖 mask）| ✗ | $O(N^2)$ 若严格 | 强 | 强（光谱惯性 + 空间 mask） |
| D1 Born 近似 | 近似 | $O(KN\log N)$ | 强 | 强 |
| D2 常数 $m^2$ | ✓ | $O(N\log N)$ | 无 | 弱（退化） |
| D\* 全局 mask 平均 | ✓ | $O(N\log N)$ | 弱（仅全局） | 中 |
| D3 光谱依赖质量 | 局部 | $O(N\log N + B^2)$ | 无 | 中（独立于 mask） |

**推荐策略**：
1. 如果论文创新强度要求高 → 用 D1 Born 近似
2. 如果要保证稳定性 → 用 D\* + 方案 A（两者结合：mask 同时做初始门控和全局质量调制）
3. D2 和 D3 不作为主力方案

---

## 5. Transformer 处理的三条路线

### 5.1 Option 1：完全舍弃 Transformer（纯 WPO + FFN）

**架构设计**：
```
Input
 ↓
LN → WPO (3D) → Residual
 ↓
LN → FFN → Residual
 ↓
Output
```

完全对应 WaveFormer 的做法，将 MST 的 S-MSA 整个模块替换为 WPO。

**利**：
1. **创新强度最大**：彻底的物理建模范式，"MST 风格的 Mask + 波动方程物理"，故事线最干净
2. **参数量最小**：省去 S-MSA 的 $Q, K, V$ 投影矩阵
3. **复杂度最低**：$O(N\log N)$ 全程保持
4. **直接对应 WaveFormer 方法学**：论文审稿人会联想 WaveFormer 的成功，容易接受
5. **故事最好讲**：一张对比图（Attention vs WPO）就能说清创新

**弊**：
1. **放弃 MST 的核心贡献**：S-MSA 的光谱注意力是 MST 被引用的主要原因，完全舍弃意味着 MST 变成纯 baseline，从名字"WaveMST"到"WaveFormer for HSI"
2. **风险高**：MST 的 S-MSA 在 31 个光谱 token 上的注意力有充分实证（3 篇顶会），突然抛弃可能性能下降
3. **3D WPO 在各向异性强的 HSI 上是否真能替代 S-MSA，缺乏先验证据**
4. **创新被归功于 WaveFormer 而非你**：审稿人可能说"这只是 WaveFormer 在 HSI 上的应用"

**适用情景**：
- 你相信波动方程的物理先验强到可以完全替代数据驱动的注意力
- 你有足够算力做大规模消融证明 WPO > S-MSA
- 你愿意承担可能性能下降的风险换取创新强度

### 5.2 Option 2：WPO 替代 S-MSA，保留 FFN（★ 中庸方案）

**架构设计**：
```
Input
 ↓
LN → WPO-Spectral-aware → Residual  (替代 S-MSA 的位置)
 ↓
LN → FFN → Residual
 ↓
Output
```

核心思路：把 MST-Block 中的 S-MSA 子块替换为一个 3D WPO 或 Spectral-WPO，其他结构（LN、残差、FFN、下采样）不变。

**利**：
1. **架构与 MST 精神一致**：仍然是"注意力+FFN"的 Transformer 范式，只是"注意力"的实现方式变为物理传播
2. **与 WaveFormer 对应最好**：WaveFormer 的做法就是"WPO 替代 Self-Attention"，直接迁移过来
3. **参数量与 MST 相当**：FFN 贡献了大部分参数
4. **容易消融**：可以逐层替换 S-MSA 看性能变化
5. **实现直接**：只需要修改 MST 代码中的 S-MSA 模块

**弊**：
1. **完全替代 S-MSA 仍有风险**：见 Option 1 的"弊"条款 2-3
2. **创新点仅"局部替代"**：没有探索 WPO 和 Attention 的互补性

**与 Option 1 的区别**：
- Option 1 是**架构级**重新设计（抛弃 Transformer 骨架）
- Option 2 是**模块级**替换（保留 Transformer 骨架，只换一个组件）
- 实际上 WaveFormer 做的就是 Option 2，不过它的 baseline 是 ViT 而非 MST

**适用情景**：
- 你想做"WaveFormer for HSI"但想加 MST 的 mask 创新
- 你希望快速跑通实验，基于 MST 代码库改最少的东西

### 5.3 Option 3：WPO 与 S-MSA 并联（★★ 我的最高推荐）

**架构设计**：
```
Input
 ↓
LN
 ├──→ WPO-3D (空-谱物理传播) ─┐
 ├──→ S-MSA (光谱注意力) ──────┤
 │                             ↓
 │                        Concat/Add + Linear
 ↓
Residual
 ↓
LN → FFN → Residual
 ↓
Output
```

两个分支并联处理同一输入，输出融合后加残差。融合方式可以是：
- **并加**：$\text{out} = \text{WPO}(x) + \text{S-MSA}(x)$（参数最少）
- **门控**：$\text{out} = g \cdot \text{WPO}(x) + (1-g) \cdot \text{S-MSA}(x)$，其中 $g$ 可学习
- **拼接+线性**：$\text{out} = W[\text{WPO}(x); \text{S-MSA}(x)]$（表达力最强）

**利**（这是我详写的部分）：

1. **双重长距离建模**：
   - **WPO**：基于物理的全局传播，**显式建模频率**，保留高频细节
   - **S-MSA**：基于数据的相似度计算，**隐式建模光谱相关性**，对同质像素强响应
   - 两者捕捉不同的信息模式：WPO 更偏结构性（波前面、振荡模态），S-MSA 更偏统计性（像素相似性）

2. **论文故事线极强**：
   - "We combine the data-driven power of attention with the physical principled propagation of waves"
   - 可以做 ablation 证明两者互补（WPO only vs S-MSA only vs 并联）
   - Figure 里可以画"WPO 的频域响应 + S-MSA 的注意力图"组合展示两种机制

3. **风险最低**：
   - 如果 WPO 无效，S-MSA 仍在工作（退化为 MST）
   - 如果 S-MSA 无效，WPO 仍在工作（退化为 Option 1/2）
   - 训练鲁棒性强

4. **消融实验最丰富**：
   - 可以做完整的 4×2 矩阵（四种 mask × 有无 S-MSA）
   - 每个 ablation 都对应一个明确的假设检验
   - 审稿人喜欢充分的消融实验

5. **参数量控制灵活**：
   - 并联两个模块看似增加参数，但实际上可以让每个模块宽度减半（WPO 和 S-MSA 各 $d/2$），总参数量可以与 MST 基线相同
   - 或者接受参数增加换取性能提升，实验数据会说话

6. **与 HSI 物理的多层次对应**：
   - **WPO**：对应 CASSI 的物理光传播过程
   - **S-MSA**：对应 HSI 光谱自相似性的数据先验
   - 两者结合 = 物理先验 + 数据先验的融合（这是 Physics-Informed ML 的核心理念）

7. **易于扩展**：如果未来想加第三个分支（如 Mamba），架构可以直接扩展为三路并联

**弊**：
1. **架构稍复杂**：三个子模块（WPO、S-MSA、FFN）比 Option 1/2 的两个子模块略复杂
2. **参数量可能增加**：如果不缩减宽度，参数量约增加 30-50%
3. **训练可能需要更多调参**：两个分支的权重可能需要 warmup 或分别调学习率
4. **"并联"本身不算最惊艳的创新**：需要在 mask 方案或 WPO 改进上补强创新点

**论文写作策略**：
- **标题**：强调 physics-informed 和 data-driven 的结合，例如 "Wave-MST: Physics-Informed Spectral Wave Propagation with Mask-Guided Attention"
- **故事**：先讲 HSI 的两个基本先验（物理+统计），再讲单一方法的局限（MST 只用统计、WaveFormer 只用物理），最后讲两者融合的必要性
- **Contributions**：（1）首个将波动方程引入光谱重建；（2）Mask-guided WPO 设计；（3）物理-数据融合架构

**适用情景**：
- 稳妥优先，想确保论文能中
- 希望得到可靠的性能提升（而非理论最优但实验翻车）
- 计划做充分的消融实验

**为什么我最推荐 Option 3**：在科研中，"稳妥+创新"比"激进+风险"更适合作为主力方案。Option 3 提供了"baseline 保底 + 创新上限"，而 Option 1/2 是"要么全赢要么全输"。WaveFormer 敢做 Option 2 是因为它的对标是 ViT（baseline 相对弱），MST 作为强 baseline，直接替换 S-MSA 面临的风险更大。

### 5.4 三种 Option 总结比较

| 特性 | Option 1（纯 WPO）| Option 2（替代）| Option 3（并联）|
|------|-----------------|---------------|---------------|
| 创新强度 | 极高 | 高 | 中高 |
| 实现难度 | 中 | 低 | 中 |
| 参数量 vs MST | $-30\%$ | $\approx$ | $+30\sim50\%$ |
| 风险 | 高 | 中 | 低 |
| 故事线 | "物理取代数据" | "波代替注意力" | "物理+数据融合" |
| 消融实验 | 少 | 中 | 多 |
| 论文接受概率（主观） | 中（有争议性） | 中高 | 高 |
| 与 MST 精神一致性 | 低 | 中 | 高 |

---

