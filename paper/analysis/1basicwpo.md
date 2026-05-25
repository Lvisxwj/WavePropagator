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

### 3.5 HSI 任务适配度分析

**支持路线 3 的理由**：
1. HSI 的核心特性"same material, similar spectra"是**空谱本质耦合**——空间相邻像素倾向于同时在多个波段相关，3D 波动自然建模这种耦合
2. CASSI 的物理测量也是空谱耦合的（每个空间位置的光谱被 mask 编码后叠加到 2D 传感器上）
3. 模型与物理更同构

**最终判决**：路线 3（更自然的物理建模，更有故事）
- **路线 2** 作为消融对照组（"我们发现耦合比解耦好，提升了 X%"）

---

## 4. Mask 方案数学分析

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

完全对应 WaveFormer 的做法

**利**：
1. **创新强度最大**：彻底的物理建模范式，"MST 风格的 Mask + 波动方程物理"，故事线最干净
2. **参数量最小**：省去 S-MSA 的 $Q, K, V$ 投影矩阵
3. **复杂度最低**：$O(N\log N)$ 全程保持
4. **直接对应 WaveFormer 方法学**：论文审稿人会联想 WaveFormer 的成功，容易接受
5. **故事最好讲**：一张对比图（Attention vs WPO）就能说清创新
