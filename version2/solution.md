# Version 2 NaN 问题诊断与修复方案

> **症状**：训练出现大量 NaN，甚至负 PSNR。
> **根因**：经过逐文件对比，发现 **3 个确定 bug** 和 **2 个高风险点**。

---

## 1. 确定 Bug

### Bug 1（致命）：`z_clean = z * deg_weight` 删掉了残差连接

**位置**：`model/unfolding.py` 第 193 行

```python
# 当前代码（错误）：
z_clean = z * deg_weight

# handoff 中写的（正确）：
z_clean = z * deg_weight + z
```

`deg_weight` 是 Sigmoid 输出，范围 [0, 1]。直接 `z * deg_weight` 会把特征值**压缩到原来的 0~1 倍**。初始训练时 deg_weight ≈ 0.5（Sigmoid 初始化），等于直接把信号砍半。经过 5 个 stage 累乘：$0.5^5 = 0.03$——信号衰减到 3%，梯度消失，最终 NaN。

**修复**：

```python
z_clean = z * deg_weight + z    # 净化 + 残差保底
```

或更安全的写法：

```python
z_clean = z * (1.0 + deg_weight)   # 不会衰减，只会增强 1~2 倍
```

### Bug 2（致命）：输出残差缺少 GD step 的 `z`

**位置**：`model/unfolding.py` 第 204 行

```python
# 当前代码（错误）：
f = f_wave + f_local

# handoff 中写的（正确）：
f = z + f_wave + f_local
```

`f_wave` 是 WaveMST_3D 的输出——它内部有全局残差 `self.mapping(fea) + x`，所以 `f_wave` 已经包含了 `z_clean` 的信息。但 `z` ≠ `z_clean`（差一个 deg_weight），加上 `z` 才能保证数据保真步（GD step）的结果被完整保留。

不加 `z` 的后果：GD step 的修正信息被丢弃，unfolding 退化为纯 prior network 串联——失去了 unfolding 的核心意义。

**修复**：

```python
f = z + f_wave + f_local
```

### Bug 3（重要）：共享权重模式下 sigma 估计在循环外只调了一次

**位置**：`model/unfolding.py` 第 149-162 行

```python
# 当前代码：
if self.share_weights:
    deg_est = self._get_deg_est(0)
    delta_Phi_pre, deg_weight_pre, _ = deg_est(f, Phi, Phi_star)  # 循环外调用
    ...

for k in range(self.num_stages):
    if self.share_weights:
        sigma = self._get_deg_est(0).sigma_est(f).view(-1, 1, 1, 1)  # 每轮更新 sigma
        deg_weight = deg_weight_pre    # ← 但 deg_weight 始终用初始的 f 算出来的！
```

问题：`deg_weight_pre` 是在 `f = initial_conv(...)` 时算的——此时 f 是最粗糙的初始估计。后续 stage 中 f 越来越好，但 deg_weight 始终用最差的 f 对应的权重。

**修复**：在循环内每次重新算 deg_weight（sigma 已经在循环内更新了）：

```python
for k in range(self.num_stages):
    if self.share_weights:
        _, deg_weight, sigma = self._get_deg_est(0)(f, Phi, Phi_star)
        if self.use_ahqs:
            delta_Phi = self._get_deg_est(0).delta_phi(Phi)
            Phi_eff_shift = shift_batch(Phi + delta_Phi, self.len_shift)
    else:
        delta_Phi, deg_weight, sigma = self._get_deg_est(k)(f, Phi, Phi_star)
        if self.use_ahqs:
            Phi_eff_shift = shift_batch(Phi + delta_Phi, self.len_shift)
```

---

## 2. 高风险点

### 风险 1：`ParaEstimator` 输出无约束——可能产生极大步长

**位置**：`model/utils.py` ParaEstimator

```python
def forward(self, x):
    ...
    x = self.mlp(x) + self.bias    # bias 初始化为 1.0
    return x                        # 没有 sigmoid 或 clamp！
```

stage2 原版也是这样，训练正常。但 version2 加了 `delta_Phi` 修正 Phi，GD step 的数值行为可能变化。如果 `rho_k` 太大（比如 10+），GD step 产生极大更新，导致 NaN。

**建议修复**（安全但不改默认行为）：在 unfolding.py 的 GD step 之后加 clamp：

```python
z = f + rho_k * mul_PhiT_residual(...)
z = z.clamp(min=-1.0, max=2.0)   # HSI 值域 [0,1]，留余量
```

### 风险 2：`construct_degraded_mask` 的输出可能有极端值

**位置**：`model/degradation.py` 第 29 行

```python
Phi_star = 2.0 * Phi_star / C     # C=28，归一化
```

如果 Phi 的值不在 [0,1]（比如某些 mask 实现中 Phi ∈ {0, 1}），`Phi_star` 的值可能在 [0, 2]——和 Phi 的 [0, 1] 不在同一量级。传入 `deg_weight` 估计网络时 `cat([Phi, Phi_star])`，量级不一致可能导致训练不稳定。

**建议修复**：

```python
Phi_star = Phi_star / (Phi_star.max() + 1e-6)  # 归一化到 [0, 1]
```

---

## 3. 修复后的 unfolding.py 核心循环

```python
def forward(self, g, input_mask):
    Phi, PhiPhiT = input_mask
    Phi_shift = shift_batch(Phi, self.len_shift)
    
    # 预计算退化 mask（只算一次）
    Phi_star = construct_degraded_mask(Phi, self.len_shift)
    
    # 初始化
    g_normal = g / self.nC * 2
    temp_g = g_normal.repeat(1, self.nC, 1, 1)
    f0 = shift_back_batch(temp_g, self.len_shift, self.size)
    f = self.initial_conv(torch.cat([f0, Phi], dim=1))
    
    if self.use_ahqs:
        f_prev = f.detach().clone()
    
    outputs = []
    
    for k in range(self.num_stages):
        # 1. 退化估计（每个 stage 都重新算）
        deg_est = self._get_deg_est(k)
        delta_Phi, deg_weight, sigma = deg_est(f, Phi, Phi_star)
        
        if self.use_ahqs:
            # 2. Nesterov 动量
            beta_k = torch.sigmoid(self.betas[k])
            f_momentum = f + beta_k * (f - f_prev)
            f_prev = f.detach().clone()
            
            # 3. 修正 GD step
            rho_k = self.rho_estimators[k](f_momentum)
            Phi_eff = Phi + delta_Phi
            Phi_eff_shift = shift_batch(Phi_eff, self.len_shift)
            Phi_f = mul_Phi_f(Phi_eff_shift, f_momentum, self.len_shift)
            residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
            residual = residual.clamp(min=-10, max=10)
            z = f_momentum + rho_k * mul_PhiT_residual(
                Phi_eff_shift, residual, self.len_shift, self.size
            )
        else:
            # GAP 标准路径
            rho_k = self.rho_estimators[k](f)
            Phi_f = mul_Phi_f(Phi_shift, f, self.len_shift)
            residual = (g - Phi_f) / PhiPhiT.clamp(min=1e-6)
            residual = residual.clamp(min=-10, max=10)
            z = f + rho_k * mul_PhiT_residual(
                Phi_shift, residual, self.len_shift, self.size
            )
        
        # 4. 初始场净化（★ 关键修复：加残差）
        z_clean = z * deg_weight + z
        
        # 5. WPO 传播
        prior = self._get_prior(k)
        f_wave = prior(z_clean, Phi, sigma=sigma)
        
        # 6. 局部精化
        refine = self._get_refine(k)
        f_local = refine(f_wave)
        
        # 7. 输出（★ 关键修复：加 z 残差）
        f = z + f_wave + f_local
        
        outputs.append(f)
    
    return outputs
```

---

## 4. Debug 建议

在修复上述 bug 后，如果仍有问题，在 `unfolding.py` 的 forward 循环中加以下 debug 打印（训练几个 batch 后关闭）：

```python
if k == 0 and outputs == []:  # 只打印第一个 stage 的第一个 batch
    print(f"[DEBUG] z       : min={z.min():.4f} max={z.max():.4f} mean={z.mean():.4f}")
    print(f"[DEBUG] deg_w   : min={deg_weight.min():.4f} max={deg_weight.max():.4f}")
    print(f"[DEBUG] sigma   : {sigma.mean():.4f}")
    print(f"[DEBUG] z_clean : min={z_clean.min():.4f} max={z_clean.max():.4f}")
    print(f"[DEBUG] f_wave  : min={f_wave.min():.4f} max={f_wave.max():.4f}")
    print(f"[DEBUG] rho_k   : {rho_k.mean():.4f}")
    if self.use_ahqs:
        print(f"[DEBUG] beta_k  : {torch.sigmoid(self.betas[k]):.4f}")
```

如果 `z` 的值已经是 NaN → 问题在 GD step 之前（rho_k 太大或 PhiPhiT 有问题）。

如果 `z_clean` 是 NaN 但 `z` 正常 → 问题在 deg_weight（Phi_star 计算有误）。

如果 `f_wave` 是 NaN 但 `z_clean` 正常 → 问题在 WPO 内部（alpha_eff 因 sigma 过大导致 exp 溢出）。

---

## 5. 首次运行建议

修复 bug 后，先用**最简配置**验证能跑通：

```python
# __init__.py 设置为最保守配置
USE_KG = False
WPO_FBGW_MODE = 'none'
USE_SWIN_WPO = False
USE_UNFOLDING = True
USE_AHQS = False         # ★ 先不开 A-HQS
NUM_STAGES = 5
SHARE_STAGE_WEIGHTS = True
```

这个配置和 stage2 的纯 WPO 5stg 几乎一致，只多了 DegradationEstimation 和 LocalRefinement。如果这个配置能正常训练（10 epoch 后 PSNR > 33），说明基础框架没问题，再逐个开启新功能。

