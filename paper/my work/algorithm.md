# algorithm.md — SMILE² 完整数学推导 + 工程映射

> 本文件分两大部分：
>
> - **Part 1（数学）**：从基础方程开始的严格推导。任何 “省略一行就讲不通” 的步骤都补齐。这里只关心数学正确性，不关心 PyTorch 实现。
> - **Part 2（工程）**：把代码里的每个张量/算子精确对应到 Part 1 的公式编号上。读者读完一段代码可以反查公式编号验证一致性。
>
> 所有命名严格遵循 `name_mapping.md`：SMILE² / SWAP / MI / AdaSpec / KGD / W-SWAP / LDE / SEC / DAG / NLE / LRB / A-HQS。

---

# Part 1 — 数学推导

## 1.1 记号约定

- 连续坐标 $\mathbf r = (x, y, \lambda) \in [0, H]\times[0, W]\times[1, \Lambda]$，时间 $t \ge 0$。
- 待求场 $u(\mathbf r, t) \in \mathbb R$，初始振幅 $u_0(\mathbf r) = u(\mathbf r, 0)$，初始速度 $v_0(\mathbf r) = \partial_t u\big|_{t=0}$。
- 3D Fourier 变换 $\hat u(\boldsymbol\omega, t) = \int u(\mathbf r, t)\, e^{-i\boldsymbol\omega\cdot\mathbf r}\, d\mathbf r$，$\boldsymbol\omega = (\omega_x, \omega_y, \omega_\lambda)$。
- 频域微分性质：$\mathcal F[\partial_x^2 u] = -\omega_x^2 \hat u$，$\mathcal F[\partial_t^n u] = \partial_t^n \hat u$（Fourier 变换与时间求导对易）。

## 1.2 各向异性 3D 阻尼波动方程（SWAP 的基础）

物理上 HSI 立方体的空间与光谱尺度差异巨大（$H,W{\sim}256$ vs $\Lambda{\sim}28$），因此假设各向异性的波速：

$$
\boxed{
\partial_t^2 u + \alpha\,\partial_t u
\;=\;
v_s^2\bigl(\partial_x^2 u + \partial_y^2 u\bigr) + v_\lambda^2\,\partial_\lambda^2 u
}
\tag{1.1}
$$

参数：

- $\alpha > 0$ 为阻尼系数（控制能量耗散速率）；
- $v_s > 0$ 为空间波速；
- $v_\lambda > 0$ 为光谱波速；
- 初始条件 $u(\mathbf r, 0) = u_0(\mathbf r),\; \partial_t u\big|_{t=0} = v_0(\mathbf r)$。

各向同性退化情形（$v_s = v_\lambda$）作为消融。

## 1.3 频域 ODE

对 (1.1) 两端做 3D Fourier 变换：

$$
\partial_t^2 \hat u + \alpha \partial_t \hat u + \omega_0^2(\boldsymbol\omega)\,\hat u = 0,
\tag{1.2}
$$

其中**固有频率**

$$
\boxed{
\omega_0^2(\boldsymbol\omega)
\;=\;
v_s^2\bigl(\omega_x^2 + \omega_y^2\bigr) + v_\lambda^2\,\omega_\lambda^2.
}
\tag{1.3}
$$

每个 $\boldsymbol\omega$ 点是独立的二阶常系数线性 ODE。**所有空间-光谱频率解耦**。

## 1.4 特征方程与判别式

ansatz $\hat u \propto e^{rt}$ 代入 (1.2) 得

$$r^2 + \alpha r + \omega_0^2 = 0, \qquad r_\pm = -\frac{\alpha}{2} \pm \sqrt{\frac{\alpha^2}{4} - \omega_0^2}.
\tag{1.4}$$

定义判别量

$$
\eta(\boldsymbol\omega) \;:=\; \omega_0^2 - \frac{\alpha^2}{4}.
\tag{1.5}
$$

三种行为：

- 欠阻尼 $\eta > 0$：$r_\pm = -\frac{\alpha}{2} \pm i\,\omega_d,\; \omega_d := \sqrt{\eta} > 0$，频域振荡式衰减。
- 过阻尼 $\eta < 0$：$r_\pm = -\frac{\alpha}{2} \pm \gamma,\; \gamma := \sqrt{-\eta} > 0$，纯指数衰减。
- 临界阻尼 $\eta = 0$：重根，零测度集，可忽略。

## 1.5 欠阻尼闭式解

通解 $\hat u(\boldsymbol\omega, t) = e^{-\alpha t/2}\bigl[A\cos(\omega_d t) + B\sin(\omega_d t)\bigr]$。

应用初值条件：

- $\hat u(\boldsymbol\omega, 0) = \hat u_0 \Rightarrow A = \hat u_0$。
- $\partial_t \hat u\bigl|_{t=0} = -\frac{\alpha}{2}\hat u_0 + \omega_d B = \hat v_0$，解出 $B = \dfrac{\hat v_0 + \tfrac{\alpha}{2}\hat u_0}{\omega_d}$。

**欠阻尼解**

$$
\boxed{
\hat u(\boldsymbol\omega, t)
\;=\;
e^{-\alpha t/2}\!\left[\hat u_0\cos(\omega_d t) + \frac{\hat v_0 + \tfrac{\alpha}{2}\hat u_0}{\omega_d}\sin(\omega_d t)\right]
}
\tag{1.6}
$$

## 1.6 过阻尼闭式解

通解 $\hat u = e^{-\alpha t/2}\bigl[A\cosh(\gamma t) + B\sinh(\gamma t)\bigr]$。同样代入初值：

$$
\boxed{
\hat u(\boldsymbol\omega, t)
\;=\;
e^{-\alpha t/2}\!\left[\hat u_0\cosh(\gamma t) + \frac{\hat v_0 + \tfrac{\alpha}{2}\hat u_0}{\gamma}\sinh(\gamma t)\right]
}
\tag{1.7}
$$

## 1.7 统一闭式解（用 $\eta$ 的符号）

定义符号自适应的“广义 $\mathrm{Cs}, \mathrm{Sn}$”：

$$
\mathrm{Cs}(\eta, t)
=
\begin{cases}
\cos(\sqrt{\eta}\,t), & \eta > 0,\\
\cosh(\sqrt{-\eta}\,t), & \eta < 0,
\end{cases}
\qquad
\mathrm{Sn}(\eta, t)
=
\begin{cases}
\dfrac{\sin(\sqrt{\eta}\,t)}{\sqrt{\eta}}, & \eta > 0,\\[6pt]
\dfrac{\sinh(\sqrt{-\eta}\,t)}{\sqrt{-\eta}}, & \eta < 0.
\end{cases}
\tag{1.8}
$$

**SWAP 统一闭式解**

$$
\boxed{
\hat u(\boldsymbol\omega, t)
\;=\;
\underbrace{e^{-\alpha t/2}}_{\text{decay}}
\,\Big[
\hat u_0\,\mathrm{Cs}(\eta, t) \;+\;
\bigl(\hat v_0 + \tfrac{\alpha}{2}\hat u_0\bigr)\,\mathrm{Sn}(\eta, t)
\Big]
}
\tag{1.9}
$$

> **稳定性观察**：$\mathrm{Sn}$ 在 $\eta\to 0$ 时是良定义的（$\mathrm{Sn}(0, t) = t$，由 $\sin x / x \to 1, \sinh x / x \to 1$ 推出），因此整个表达式在临界阻尼点连续。WaveFormer 原文仅给 (1.6)，当 $\eta < 0$ 时 `sqrt(omega_d_sq)` 会产生 NaN —— SMILE² 的工程实现必须按 (1.9) 处理两种分支。

## 1.8 能量泛函与稳定性

频域能量密度

$$
E(\boldsymbol\omega, t)
= \tfrac{1}{2}\big|\partial_t \hat u\big|^2
+ \tfrac{1}{2}\,\omega_0^2\,|\hat u|^2.
\tag{1.10}
$$

对 $t$ 求导，代入 (1.2)：

$$
\frac{dE}{dt}
= \mathrm{Re}\bigl[\partial_t\hat u^* \cdot \partial_t^2 \hat u\bigr] + \omega_0^2\,\mathrm{Re}[\hat u^*\partial_t \hat u]
= \mathrm{Re}\bigl[\partial_t\hat u^*(-\alpha\partial_t\hat u - \omega_0^2 \hat u + \omega_0^2 \hat u)\bigr]
= -\alpha\,|\partial_t\hat u|^2 \le 0.
\tag{1.11}
$$

**结论**：能量单调不增 ⇒ SWAP 的频域调制是无条件稳定的，与 $\alpha, v_s, v_\lambda, t$ 的取值无关。这是不需要梯度裁剪的物理基础。

## 1.9 Modulated Initialization (MI)：Mask 软门控

CASSI 的 mask 在物理过程 (1.1) 中对应**只对初始振幅施加一次空间调制**：

$$
u_0^{\mathrm{masked}}(x, y, \lambda)
\;=\;
\bigl[\epsilon + (1-\epsilon)\,M(x, y)\bigr]\cdot u_0(x, y, \lambda),
\qquad \epsilon \in [0, 0.2].
\tag{1.12}
$$

$\epsilon$ 是软门控下限，避免完全遮蔽导致的零梯度。等价地，速度场 $v_0$ 也乘相同门控（与 MI 对应的 `phi/psi` 同步）。

### 1.9.1 闭式解保持性

由卷积定理：

$$
\mathcal F\bigl[M \cdot u_0\bigr](\boldsymbol\omega)
= \hat M(\omega_x, \omega_y)\,*_{xy}\,\hat u_0(\boldsymbol\omega),
\tag{1.13}
$$

其中 $*_{xy}$ 仅在空间频率上做 2D 卷积（光谱方向上 $M$ 是常数，$\hat M$ 在 $\omega_\lambda$ 维度退化为 $\delta(\omega_\lambda)$）。

把 (1.13) 代入 (1.9)：

$$
\hat u(\boldsymbol\omega, t)
= e^{-\alpha t/2}\Bigl[(\hat M *_{xy} \hat u_0)\,\mathrm{Cs}(\eta, t)
+ \bigl(\hat v_0 + \tfrac{\alpha}{2}\hat M *_{xy} \hat u_0\bigr)\,\mathrm{Sn}(\eta, t)\Bigr].
\tag{1.14}
$$

**复杂度保持**：实现上不必显式做卷积——在空间域乘 $M$ 再做 FFT 即可，总复杂度仍为 $O(N\log N)$。

### 1.9.2 能量损失上界

由 Parseval 定理，

$$
\|u_0^{\mathrm{masked}}\|_2^2
= \sum_{xy\lambda} \bigl[\epsilon + (1-\epsilon)M\bigr]^2 u_0^2
\le \|u_0\|_2^2.
\tag{1.15}
$$

CASSI 中典型 mask 透射率 $\bar M \approx 0.5$，能量保留比例 $\eta_M \approx 0.5(1+\epsilon)^2 \approx 0.6$。

## 1.10 Adaptive Spectral Filtering (AdaSpec)：Wiener 嵌入

在频域调制 (1.9) 之后，再乘一项 SNR 自适应权重：

$$
\boxed{
W_{\mathrm{AdaSpec}}(\boldsymbol\omega)
\;=\;
\sigma_{\mathrm{gate}}\!\left(\frac{|\hat u_0(\boldsymbol\omega)|^2 - \sigma^2}{|\hat u_0(\boldsymbol\omega)|^2 + \sigma^2 + \epsilon}\right) \in [0, 1]
}
\tag{1.16}
$$

其中 $\sigma_{\mathrm{gate}}(\cdot) = \mathrm{sigmoid}(\cdot)$。**直观**：信号频带（$|\hat u_0|^2 \gg \sigma^2$）→ 比值接近 $+1$，$W \to 1$（保留）；噪声频带（$|\hat u_0|^2 \approx \sigma^2$）→ 比值接近 $0$，$W \to 0.5$（中性）；纯噪声频带（$|\hat u_0|^2 \ll \sigma^2$）→ 比值接近 $-1$，$W \to 0$（抑制）。

### 1.10.1 与 Wiener 滤波器的关系

经典 Wiener 滤波器形式

$$
W_{\mathrm{Wiener}}(\boldsymbol\omega) = \frac{|\hat u_0|^2}{|\hat u_0|^2 + \sigma^2}.
\tag{1.17}
$$

AdaSpec 把上式重写为 $W = \mathrm{sigmoid}\bigl((|\hat u_0|^2 - \sigma^2)/(|\hat u_0|^2 + \sigma^2 + \epsilon)\bigr)$ —— 在 $|\hat u_0|^2 \gg \sigma^2$ 与 $|\hat u_0|^2 \ll \sigma^2$ 两端与 Wiener 同极限；并且**零额外参数**（$\sigma$ 复用 NLE 输出）。

### 1.10.2 可学习频带变体（学习版）

将频率体按 $|\boldsymbol\omega|$ 分成 $K$ 个频带，每个频带分配一个可学权重 $w_k = \mathrm{softplus}(\theta_k)$：

$$
W_{\mathrm{AdaSpec}}^{\mathrm{learn}}(\boldsymbol\omega)
= w_{\mathrm{band}(|\boldsymbol\omega|)}.
\tag{1.18}
$$

总参数 $K$ 个标量（默认 $K{=}8$）。论文里 (1.16) 为主，(1.18) 作消融。

## 1.11 Klein-Gordon Dispersion (KGD)：可选光谱硬先验

将 (1.1) 推广为带质量项的 Klein-Gordon 方程：

$$
\partial_t^2 u + \alpha\,\partial_t u
= v_s^2(\partial_x^2 + \partial_y^2) u + v_\lambda^2\,\partial_\lambda^2 u - m^2(x, y)\,u,
\tag{1.19}
$$

其中

$$
m^2(x, y) \;=\; m_0^2 \cdot \bigl[1 - M(x, y)\bigr],\quad m_0^2 \in [0, 0.5].
\tag{1.20}
$$

> 物理直觉：mask 透射率低 ⇒ $m^2$ 大 ⇒ 强抑制；透射率高 ⇒ $m^2 \to 0$，退化为普通波方程。

### 1.11.1 频域 ODE 的耦合

由卷积定理，

$$
\mathcal F\bigl[m^2(x, y)\,u(\mathbf r, t)\bigr]
= \hat{m^2}(\omega_x, \omega_y)\,*_{xy}\,\hat u(\boldsymbol\omega, t).
\tag{1.21}
$$

(1.19) 的频域方程因此变为

$$
\partial_t^2 \hat u + \alpha\,\partial_t\hat u + \omega_0^2\,\hat u
+ \bigl(\hat{m^2}\,*_{xy}\,\hat u\bigr) = 0.
\tag{1.22}
$$

**不再是逐点 ODE**，所有空间频率通过卷积耦合 → 不可分。直接闭式解 $O(N^2)$。

### 1.11.2 Born 一阶微扰

把 $-m^2 u$ 视作微扰，令 $u = u^{(0)} + u^{(1)} + O(m_0^4)$。**零阶**：$u^{(0)}$ 满足 (1.1)，闭式解给出 (1.9)。**一阶修正**：

$$
\partial_t^2 u^{(1)} + \alpha\partial_t u^{(1)}
- v_s^2 \nabla_{xy}^2 u^{(1)} - v_\lambda^2 \partial_\lambda^2 u^{(1)}
= -\,m^2(x, y)\,u^{(0)}(\mathbf r, t).
\tag{1.23}
$$

Duhamel 积分给出

$$
\hat u^{(1)}(\boldsymbol\omega, t)
= -\int_0^t G(\boldsymbol\omega, t - \tau)\, \mathcal F\bigl[m^2 u^{(0)}\bigr](\boldsymbol\omega, \tau)\,d\tau,
\tag{1.24}
$$

其中 Green 函数 $G(\boldsymbol\omega, t) = e^{-\alpha t/2}\,\mathrm{Sn}(\eta, t)$。

#### 单步 Born 简化（SMILE² 中的实际使用）

在我们工程中只取 **单一 $\tau$ = $t$** 的 Born 近似：

$$
\hat u^{(1)}(\boldsymbol\omega, t)
\approx
-\,e^{-\alpha t/2}\,\mathrm{Sn}(\eta, t)\,\mathcal F\bigl[m^2\,u^{(0)}\bigr](\boldsymbol\omega).
\tag{1.25}
$$

实现上：
1. 空间域计算 $u^{(0)}$（已由 SWAP 主路径给出）；
2. 空间乘法 $s = -m^2\,u^{(0)}$（$O(N)$）；
3. 3D rFFT、乘 $\mathrm{Sn}(\eta, t)\,e^{-\alpha t/2}$、3D irFFT，得到 $u^{(1)}$；
4. 输出 $u = u^{(0)} + w_{\mathrm{KG}}\,u^{(1)}$，$w_{\mathrm{KG}}$ 为可学习标量。

误差量级 $O\bigl((m_0^2 t / \omega_0^2)^2\bigr)$，由 (1.20) 中 $m_0^2 \le 0.5$ 与 $t \le t_{\max}$ 控制。

## 1.12 Windowed SWAP (W-SWAP)：可选 Swin 改造

将 $(H, W)$ 切成大小 $M{\times}M$ 的窗（默认 $M = 64$），每个窗内独立做 (1.9)：

$$
u_{[i, j]} := \mathrm{Window}_{i, j}(u),\quad u_{[i, j]}(\mathbf r, t) = \mathrm{SWAP}\bigl(u_{[i, j]}(\mathbf r, 0)\bigr).
\tag{1.26}
$$

shifted-window（奇数层位移 $M/2$）保证跨窗信息流动。复杂度：

$$
\mathrm{SWAP}: O\bigl(\Lambda HW\log(\Lambda HW)\bigr),\quad
\mathrm{W\text{-}SWAP}: O\bigl(\Lambda HW\log(\Lambda M^2)\bigr).
\tag{1.27}
$$

W-SWAP 的有效感受野是窗内 $M$ 像素；论文中作为长边场景的可选项。

## 1.13 Learned Degradation Estimator (LDE)

LDE 输出三件套：

### 1.13.1 SEC（Sensing Error Correction）

$$
\boxed{
\Delta\Phi(\mathbf r) \;=\; W_2 \, \mathrm{LReLU}\bigl(W_1\,\Phi(\mathbf r)\bigr),\quad
W_1, W_2 \in \mathbb R^{\Lambda \times \Lambda}\text{ (1×1 conv)}.
}
\tag{1.28}
$$

修正 sensing 矩阵：$\Phi_{\mathrm{eff}} = \Phi + \Delta\Phi$。在 GD step 中替代 $\Phi$（详见 §1.15）。

### 1.13.2 退化 mask 构造

仅依赖 $\Phi$，可预计算：

1. **Shift**：$\Phi^{\mathrm{sh}}_c[x, y] = \Phi_c[x - c\cdot d, y]$，$d = \mathrm{step}$。
2. **Compress**：$\Phi^{\mathrm{cp}}[x, y] = \sum_c \Phi^{\mathrm{sh}}_c[x, y]$（2D 加和）。
3. **Reverse**：$\Phi^*_c[x, y] = \Phi^{\mathrm{cp}}[x + c\cdot d, y]$（broadcast 回 $\Lambda$ 维）。
4. **Normalize**：$\Phi^* \leftarrow 2\Phi^* / \Lambda$。

形式上 $\Phi^*$ 编码了 “mask × shift × compression” 的全部退化模式。

### 1.13.3 DAG（Degradation-Aware Gating）

$$
\boxed{
w(\mathbf r) \;=\; \sigma_{\mathrm{gate}}\!\Bigl(U_2\,\mathrm{LReLU}\bigl(U_1\,[\,\Phi\;\Vert\;\Phi^*\,]\bigr)\Bigr) \in (0, 1)^{\Lambda},
}
\tag{1.29}
$$

其中 $[\cdot\Vert\cdot]$ 表沿通道拼接，$U_1: 2\Lambda\to h$、$U_2: h\to\Lambda$（默认 $h{=}32$）。

### 1.13.4 NLE（Noise Level Estimator）

从当前估计 $f$ 中提取全局噪声水平：

$$
\boxed{
\sigma \;=\; \mathrm{softplus}\!\Bigl(\,V_2\,\mathrm{ReLU}\bigl(V_1\,\mathrm{GAP}(f)\bigr)\Bigr) \in \mathbb R_{>0},
}
\tag{1.30}
$$

$\mathrm{GAP}$ 是全局平均池化，$V_1: \Lambda\to h$，$V_2: h\to 1$。

### 1.13.5 LDE 参数量

$\Delta\Phi: 2\Lambda^2 \approx 1568$；DAG: $2\Lambda h + h\Lambda \approx 2688$；NLE: $\Lambda h + h \approx 928$。
总计 $\approx 5.2\,\mathrm{K}$（$\Lambda = 28,\,h = 32$）。

## 1.14 Local Refinement Block (LRB)

定义

$$
\boxed{
\mathrm{LRB}(x) \;=\; C_3\bigl(\mathrm{DW}_{3\times 3}\bigl(\mathrm{GELU}(C_1 x)\bigr)\bigr),
}
\tag{1.31}
$$

其中 $C_1: \Lambda\to 2\Lambda$（1×1 conv），$\mathrm{DW}_{3\times 3}: 2\Lambda\to 2\Lambda$（depth-wise 3×3 + GELU），$C_3: 2\Lambda\to\Lambda$（1×1 conv）。参数量 $\approx 4.7\,\mathrm{K}$。

LRB 设计目标：补全 SWAP 频域全局传播无法处理的局部纹理；不引入 attention（避免与 SWAP 频域算子功能重叠）。

## 1.15 Accelerated Half-Quadratic Splitting (A-HQS)

### 1.15.1 普通 HQS

引入辅助变量 $z = f$，把 (1.2 problem.md) 改写为

$$
\min_{f, z} \; \tfrac{1}{2}\|g - \Phi f\|^2 + \gamma\,\mathcal R(z) + \tfrac{\mu}{2}\,\|f - z\|^2.
\tag{1.32}
$$

交替最小化：

- $f$-子问题：$f^{k+1} = (\Phi^{\!\top}\Phi + \mu I)^{-1}(\Phi^{\!\top} g + \mu z^k)$，有闭式解。
- $z$-子问题：$z^{k+1} = \mathrm{prox}_{\gamma\mathcal R/\mu}(f^{k+1})$，由可学习 prior 实现。

### 1.15.2 Nesterov 加速

引入动量项

$$
\hat z^k = z^k + \beta_k\,(z^k - z^{k-1}),\quad \beta_k \in [0, 1).
\tag{1.33}
$$

替换 $f$-子问题中的 $z^k \to \hat z^k$：

$$
f^{k+1} = (\Phi^{\!\top}\Phi + \mu I)^{-1}(\Phi^{\!\top} g + \mu \hat z^k).
\tag{1.34}
$$

理论：当 $\mathcal R$ 凸光滑且 $\beta_k$ 选 $\frac{k-1}{k+2}$ 量级时，收敛速率从 $O(1/K)$ 提升至 $O(1/K^2)$（Nesterov 1983）。在 SMILE² 中 $\beta_k$ 由 sigmoid 参数化保证 $\in (0, 1)$。

### 1.15.3 GD step 闭式解（含 SEC）

利用 CASSI 的稀疏结构（$\Phi^{\!\top}\Phi$ 是对角 + shift-back），可以写成逐点形式：

$$
\boxed{
z^k \;=\; \hat z^k \;+\; \rho_k\,\Phi_{\mathrm{eff}}^{\!\top}\,\frac{g - \Phi_{\mathrm{eff}}\,\hat z^k}{\mu + \Phi_{\mathrm{eff}}\Phi_{\mathrm{eff}}^{\!\top}},
}
\quad \Phi_{\mathrm{eff}} = \Phi + \Delta\Phi.
\tag{1.35}
$$

$\rho_k = \mathrm{softplus}(\mathrm{MLP}(f))$ 是可学步长（**ParaEstimator**）；分母用预计算的 $\Phi\Phi^{\!\top}$ 加 clamp 以稳定数值。

### 1.15.4 Estimation-Evolution 耦合

在 (1.35) 之后进行：

$$
z^k_{\mathrm{clean}} = z^k \cdot \bigl(1 + w\bigr),\qquad
\alpha_{\mathrm{eff}} = \alpha + \lambda_\sigma \cdot \sigma,
\tag{1.36}
$$

把 DAG 的退化权重和 NLE 的噪声水平注入到 SWAP 的输入与参数中（“evolution” 受 “estimation” 调制）。

### 1.15.5 单 stage 完整公式（汇总）

$$
\boxed{
\begin{aligned}
& \text{LDE}: \quad \Delta\Phi, w, \sigma = \mathrm{LDE}(f^{k-1}, \Phi, \Phi^*),\\
& \text{Momentum}: \quad \hat z^{k-1} = f^{k-1} + \beta_k\,(f^{k-1} - f^{k-2}),\\
& \text{GD (closed-form)}: \quad z^k = \hat z^{k-1} + \rho_k\,(\Phi+\Delta\Phi)^{\!\top}\frac{g - (\Phi+\Delta\Phi)\,\hat z^{k-1}}{\mu + (\Phi+\Delta\Phi)(\Phi+\Delta\Phi)^{\!\top}},\\
& \text{Purify}: \quad z^k_{\mathrm{clean}} = z^k\cdot(1 + w),\\
& \text{SWAP}: \quad f^k_{\mathrm{wave}} = \mathrm{SWAP}(z^k_{\mathrm{clean}}, \Phi, \alpha_{\mathrm{eff}} = \alpha + \lambda_\sigma\sigma),\\
& \text{Refine}: \quad f^k = f^k_{\mathrm{wave}} + \mathrm{LRB}(f^k_{\mathrm{wave}}).
\end{aligned}
}
\tag{1.37}
$$

> 对应代码：`version2/model/unfolding.py: WPO_Unfold.forward`。

## 1.16 多 stage 损失

$$
\boxed{
\mathcal L = \sum_{k=1}^{K} w_k\,\mathrm{RMSE}\bigl(f^k, f_{\mathrm{GT}}\bigr),\quad
w_K = 1.0,\; w_{K-1} = 0.7,\; w_{K-2} = 0.5,\; w_{K-3} = 0.3.
}
\tag{1.38}
$$

$\mathrm{RMSE}(p, q) = \sqrt{\mathbb E[(p - q)^2]}$。该权重设计参考 DPU：让最后一个 stage 获得最强监督，同时保持中间 stage 的梯度通路。

## 1.17 SWAP 整体 U-Net（参考 MST 骨架）

记 $\mathrm{Block}_d(\cdot, \Phi) = (\cdot) + \mathrm{SWAP}_d(\mathrm{LN}(\cdot), \Phi) + \mathrm{FFN}(\mathrm{LN}(\cdot))$。

Encoder ($i = 0, \dots, S-1$)：

$$
\mathrm{fea}_i \leftarrow \mathrm{Block}_{d_i}^{N_i}(\mathrm{fea}_{i}, \Phi),\quad
\mathrm{fea}_{i+1} = \mathrm{Down}_4(\mathrm{fea}_i),\quad \Phi_{i+1} = \mathrm{sigmoid}(\mathrm{Down}_4 \Phi_i).
\tag{1.39}
$$

Bottleneck：$\mathrm{Block}_{d_S}^{N_S}$。
Decoder：$\mathrm{fea}\leftarrow \mathrm{Up}_2(\mathrm{fea});\;\mathrm{fea}\leftarrow \mathrm{Fuse}([\mathrm{fea}, \mathrm{fea}_i^{\mathrm{enc}}]);\;\mathrm{Block}_{d_i}^{N_i}$.

最终：$f_{\mathrm{wave}} = \mathrm{Conv}_{3\times 3}(\mathrm{fea}) + x$（global residual），与 (1.37) 的 $f^k_{\mathrm{wave}}$ 对应。

---

# Part 2 — 工程映射

> 把代码里的每一个张量、模块、可学参数追溯到 Part 1 的公式编号。一行代码 ↔ 至少一个 (Eq #)。

## 2.1 文件 → 公式索引

| 文件 / 类 | 数学位置 | 备注 |
|----------|----------|------|
| `model/wpo3d.py: WPO3D` | (1.9), (1.16), (1.18) | 闭式解 + AdaSpec + 学习版 FBGW |
| `model/wpo3d.py: WPO3DBlock` | (1.39) 中的 `Block` | 即 LN+SWAP+Res + LN+FFN+Res |
| `model/wpo3d.py: WaveMST_3D` | (1.39) 整体 | U-Net 三段 |
| `model/wpo3d.py: WaveMST_KG` | (1.19)–(1.25) | 启用 KGD 分支 |
| `model/mask_ops.py: MaskGateA` | (1.12) MI | 软门控 + Phi/Psi 双卷积初始化 $u_0, v_0$ |
| `model/mask_ops.py: MaskKleinGordonD` | (1.20), (1.25) | 质量场 + Born 一阶修正 |
| `model/degradation.py: DegradationEstimation` | (1.28)–(1.30) | SEC + DAG + NLE 三合一 |
| `model/degradation.py: construct_degraded_mask` | (1.13.2) 退化 mask 构造 | 仅依赖 $\Phi$，可缓存 |
| `model/refinement.py: LocalRefinement` | (1.31) LRB | $1{\times}1 \to \mathrm{DW}3{\times}3 \to 1{\times}1$ |
| `model/unfolding.py: WPO_Unfold.forward` | (1.37) 完整 stage | A-HQS / GAP 两种路径 |
| `model/utils.py: ParaEstimator` | (1.35) 的 $\rho_k$ | softplus 保证正 |
| `model/utils.py: shift_batch / mul_Phi_f / mul_PhiT_residual` | (1.35) 的 $\Phi, \Phi^{\!\top}$ 乘法 | 与 CASSI shift 物理对应 |

## 2.2 SWAP 工程对应（最关键的一节）

**（1）输入：** `x = z_clean ∈ [B, C=Λ, H, W]` 与 `mask_spatial ∈ [B, Λ, H, W]`、可选 `sigma ∈ [B, 1, 1, 1]`。

**（2）初始化两个语义场（MI）：**

```python
gate = self.eps + (1.0 - self.eps) * mask_spatial   # ↔ (1.12) 软门控
u0   = self.phi(x) * gate                           # ↔ (1.13)/(1.14) 的 u_0^{masked}
v0   = self.psi(x) * gate                           # ↔ (1.6),(1.7) 中的 v_0^{masked}
```

**这就是 “SWAP 初始化了 2 个语义场” 的精确指代**：
- `u0` 对应公式 (1.6)/(1.9) 中的 $\hat u_0$ 的空间域来源（再经 FFT 得到 $\hat u_0$）；
- `v0` 对应公式 (1.6)/(1.9) 中的 $\hat v_0$ 的空间域来源。

> 如果启用 KGD（`mask_mode='D'`），同步生成 **第 3 个场** `m_sq`（公式 (1.20)）。

**（3）3D rFFT（pad 到 2 的幂）：**

```python
u0_fft = torch.fft.rfftn(u0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
v0_fft = torch.fft.rfftn(v0, s=(C_fft, H_fft, W_fft), dim=(-3, -2, -1))
```

对应 $\hat u_0, \hat v_0$。`FFT_PAD_TO_POW2 = True` 仅是性能优化（cuFFT），不改变数学。

**（4）频域调制 = (1.9) 实现：**

```python
fc, fh, fw = self._build_freq_grid(C, H, W, device)    # 频率坐标 (ω_λ, ω_y, ω_x)
omega_sq = (2π)^2 * (vs^2 * (fh^2 + fw^2) + vl^2 * fc^2)   # ↔ (1.3)
eta = omega_sq - (alpha/2)^2                                # ↔ (1.5)

# 欠阻尼分支 ↔ (1.6)
omega_d = sqrt(max(eta, 0) + 1e-30)
cos_term      = cos(omega_d * t)
sinc_term_pos = sin(omega_d * t) / (omega_d + 1e-8)

# 过阻尼分支 ↔ (1.7)
gamma          = sqrt(max(-eta, 0) + 1e-30)
cosh_term      = cosh(gamma * t)
sinch_term_neg = sinh(gamma * t) / (gamma + 1e-8)

cs   = where(eta >= 0, cos_term,      cosh_term)     # ↔ Cs(η, t)  公式 (1.8)
sinc = where(eta >= 0, sinc_term_pos, sinch_term_neg) # ↔ Sn(η, t)  公式 (1.8)

decay   = exp(-alpha * t / 2)                        # ↔ e^{-αt/2}
out_fft = decay * (u0_fft * cs + (v0_fft + alpha/2 * u0_fft) * sinc)
                                                     # ↔ 统一闭式解 (1.9)
```

**（5）AdaSpec（频带加权）：**

```python
if fbgw_mode == 'snr_adaptive':
    power    = u0_fft.abs() ** 2
    sigma_sq = sigma.mean().item() ** 2 if sigma is not None else 0.01
    W = torch.sigmoid((power - sigma_sq) / (power + sigma_sq + 1e-6))   # ↔ (1.16)
    out_fft = out_fft * W
elif fbgw_mode == 'learnable_band':
    band_idx = quantize(|ω| / |ω|_max, K)
    W = softplus(self._band_weights)[band_idx]                          # ↔ (1.18)
    out_fft = out_fft * W
```

**（6）3D irFFT + 噪声感知阻尼：**

在 `_global_forward` 起始处：

```python
if sigma is not None:
    alpha = alpha + softplus(self._lambda_sigma) * sigma.mean()         # ↔ (1.36) α_eff
```

irFFT 之后裁剪到原 $\Lambda$ 通道（去除 pow2 pad）。

**（7）KGD 修正（仅 `mask_mode='D'`）：**

```python
source     = -m_sq * out                           # ↔ (1.25) 中 -m^2 u^(0)
source_fft = rfftn(source)
corr_fft   = source_fft * sinc * decay             # ↔ Green 函数 G = e^{-αt/2} Sn
correction = irfftn(corr_fft)
out        = out + kg_weight * correction          # ↔ (1.25), w_KG 可学
```

**（8）输出投影：**

```python
out = out_norm(out)
out = out * F.silu(x)         # gate（保留 SWAP 整体外侧 residual）
out = out_linear(out)
```

LayerNorm + SiLU gate + 1×1 投影是 MST 风格的 prior 输出头；不参与公式 (1.9) 的物理推导，等价于 prior network 的 post-processing。

## 2.3 W-SWAP 工程对应

`WPO3D._swin_forward` 把 $[B, \Lambda, H, W]$ 切成大小 $M{\times}M$ 的窗，每个窗独立调用 `_global_forward`（公式 (1.26)），再重组回来。Shifted window 通过 `swin_shift=True` 在奇数层启用，对应 $(1.26)$ 的 shift。

## 2.4 LDE 工程对应

```python
# Eq. (1.28)
delta_Phi   = self.delta_phi(Phi)                            # ΔΦ
# Eq. (1.29)
deg_weight  = self.deg_weight(torch.cat([Phi, Phi_star], 1)) # w
# Eq. (1.30)
sigma       = self.sigma_est(f).view(-1, 1, 1, 1)            # σ
```

`construct_degraded_mask(Phi)` 实现 (1.13.2) 的 shift → compress → reverse → normalize 四步。

## 2.5 A-HQS 工程对应（`unfolding.py: WPO_Unfold.forward`）

| 公式 | 代码位置 | 张量 |
|------|---------|------|
| (1.37) LDE | `self._get_deg_est(...)(f, Phi, Phi_star)` | `delta_Phi, deg_weight, sigma` |
| (1.33) Nesterov | `f_input = f + beta_k * (f - f_prev)` | `f_input` $= \hat z$ |
| (1.35) GD（含 $\Phi_{\mathrm{eff}}$）| `mul_Phi_f / mul_PhiT_residual + rho_k * ...` | `z` |
| (1.36) Purify | `z_clean = z * (1.0 + deg_weight)` | `z_clean` |
| (1.36) α_eff | `WPO3D._global_forward` 中 `alpha + lambda_sigma * sigma.mean()` | 内部覆盖 `alpha` |
| (1.37) SWAP 调用 | `self._get_prior(k)(z_clean, Phi, sigma=sigma)` | `f` (即 $f^k_{\mathrm{wave}}$) |
| (1.31) LRB | `f = f + self._get_refine(k)(f)` | `f` |
| 输出 | `outputs.append(f)` | $f^k$ |

GAP 模式（`use_ahqs=False`）：删除动量项 (1.33) 并把 $\Phi_{\mathrm{eff}} \to \Phi$；其余完全一致，等价于公式 (1.35) 的退化版本。

## 2.6 多 stage 损失工程对应

`train.py: multi_stage_loss`：

```python
def multi_stage_loss(outputs, gt):
    loss = rmse(outputs[-1], gt)            # ↔ (1.38) w_K = 1.0
    if K >= 2: loss += 0.7 * rmse(outputs[-2], gt)   # ↔ w_{K-1} = 0.7
    if K >= 3: loss += 0.5 * rmse(outputs[-3], gt)   # ↔ w_{K-2} = 0.5
    if K >= 4: loss += 0.3 * rmse(outputs[-4], gt)   # ↔ w_{K-3} = 0.3
    return loss
```

## 2.7 物理参数的初始化与软约束

| 参数 | 代码 | 初始化 | softplus / sigmoid |
|------|------|--------|---------------------|
| $\alpha$ | `WPO3D.alpha`           | 0.1 | softplus |
| $v_s$    | `WPO3D.vs`              | 1.0 | softplus |
| $v_\lambda$ | `WPO3D.vl`           | 0.5 | softplus |
| $t$      | `WPO3D.t`               | 1.0 | softplus |
| $\lambda_\sigma$ | `WPO3D._lambda_sigma` | -2.0 (→ $\approx 0.13$) | softplus |
| $\beta_k$ | `WPO_Unfold.betas[k]`   | 0.0 (→ 0.5) | sigmoid |
| $\rho_k$ | `ParaEstimator`           | bias=1.0 | softplus |
| $m_0^2$ | `MaskKleinGordonD.m0_sq`  | 0.1 | clamp [0, 0.5] |
| $w_{\mathrm{KG}}$ | `MaskKleinGordonD.kg_weight` | 0.1 | raw |
| $\epsilon_{\mathrm{MI}}$ | `MaskGateA.eps` | 0.1 | 固定常量 |

## 2.8 复杂度核对

| 模块 | 复杂度 | 备注 |
|------|--------|------|
| SWAP global | $O(\Lambda HW\log(\Lambda HW))$ | 3D rFFT 主导 |
| SWAP windowed (M=64) | $O(\Lambda HW\log(\Lambda M^2))$ | (1.27) |
| LDE | $O(\Lambda^2 HW)$ | 1×1 conv |
| LRB | $O(\Lambda HW)$ | depth-wise |
| GD (1.35) | $O(\Lambda HW)$ | 逐元素 |
| 单 stage | $O(\Lambda HW\log(\Lambda HW))$ | SWAP 决定 |
| K stages | $K\cdot$ 单 stage | $K=5$ |

## 2.9 数值稳健性手册

1. **$\Phi\Phi^{\!\top}$ 分母**：`PhiPhiT.clamp(min=1e-6)`（避免边界除零）；
2. **GD 残差**：`residual.clamp(-10, 10)`（防止极端 mask 区域爆炸）；
3. **$\omega_d / \gamma$**：(1.9) 实现中加 `+ 1e-30` 与 `+ 1e-8`，保证 `sqrt`/除法可微；
4. **AdaSpec 阈值**：`+ 1e-6` 防止 $0/0$；
5. **MI 软门控**：$\epsilon = 0.1$，禁止全零透射区域；
6. **NLE 输出**：`softplus`，保证 $\sigma > 0$；进入 $\alpha_{\mathrm{eff}}$ 前再 `softplus(\lambda_\sigma)`；
7. **多 stage 累积**：`z_clean = z * (1 + w)`（非 `z * w`），防止 $w \to 0$ 时信号被湮灭。

> 上述每条都对应 `solution.md` 中已确认或预防的 NaN 来源；在论文 supplementary 中可作为「Implementation details」节。

## 2.10 训练-推断的可重复性约定

- 全部物理参数和 LDE/LRB 参数均参与训练（无冻结）；
- 共享 / 独立 stage 权重通过 `SHARE_STAGE_WEIGHTS` 切换；
- `Phi_star` 在每个 batch 仅算一次（与 mask 一一对应）；
- FFT 维度 pad 到 2 的幂（`FFT_PAD_TO_POW2 = True`）以确保 cuFFT 行为稳定；
- 多 stage 输出列表里 `outputs[-1]` 是最终结果，前面用于辅助损失。

---

## 附录 A — 常用恒等式回顾

- 卷积定理：$\mathcal F[f g] = \hat f * \hat g,\; \mathcal F[f * g] = \hat f \hat g$。
- $\mathrm{Sn}(0, t) = t,\; \mathrm{Cs}(0, t) = 1$（用于 (1.9) 临界阻尼极限）。
- $\sigma_{\mathrm{gate}}$ 即 $\mathrm{sigmoid}$。
- Parseval：$\sum_{\mathbf r} |u|^2 = \tfrac{1}{N}\sum_{\boldsymbol\omega}|\hat u|^2$（离散）。

## 附录 B — 与 WaveFormer 原文公式的对照

| WaveFormer 原文 | SMILE² 对照 |
|----------------|-------------|
| Eq. (5) 1D 阻尼波 | (1.1) 推广到 3D 各向异性 |
| Eq. (8) 欠阻尼解 | (1.6) |
| 未给过阻尼解 | (1.7) ＋ 统一形式 (1.9) |
| 无 mask | (1.12) MI |
| 无 SNR 加权 | (1.16) AdaSpec |
| 无展开 / 退化估计 | (1.37) A-HQS + LDE |
