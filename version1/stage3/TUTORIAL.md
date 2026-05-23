# Stage 2: Deep Unfolding WaveMST — 使用教程

## 1. 概述

Stage 2 将 Stage 1 的 WaveMST_3D / WaveMST_KG 升级为 **Deep Unfolding** 框架，预期 PSNR 从 34.7 dB 提升到 37–38+ dB。

核心思路：将单次 forward pass 改为 K 次迭代，每次迭代包含：
1. **GD step（数据保真）**：用测量值 $g$ 约束当前估计，使重建结果与实际观测一致
2. **Prior step（物理先验）**：用 WPO3D/KG 作为可学习去噪器，施加物理约束

原有 WPO 模块（`wpo3d.py`）**完全不修改**，只在外部加 unfolding 包装。

---

## 2. 文件清单

```
stage2/
├── unfolding_ops.py     ← 新增：shift/Phi 操作 + ParaEstimator
├── wpo3d_unfold.py      ← 新增：Model 7/8 unfolding 包装类
├── dataset.py           ← 修改：增加 gen_meas_unfolding()
├── train.py             ← 修改：unfolding 配置 + 多 stage 损失
├── test.py              ← 修改：unfolding 推理 + stage-wise PSNR
├── wpo3d.py             ← 复制自 stage1，不修改
├── mask_ops.py          ← 复制自 stage1，不修改
├── physics.py           ← 复制自 stage1，不修改
├── loss.py              ← 复制自 stage1，不修改
├── DPU/                 ← 参考代码（不运行）
├── SSR/                 ← 参考代码（不运行）
└── result/              ← 训练输出（自动创建）
```

---

## 3. 模型索引

| MODEL_INDEX | 模型名 | 说明 | 参数量（5stg） |
|:-----------:|--------|------|:-------------:|
| 0 | WaveMST_3D | 原始 3D-WPO（非 unfolding） | 0.79M |
| 1 | WaveMST_KG | Klein-Gordon（非 unfolding） | 0.79M |
| **7** | **WaveMST_3D_Unfold** | **3D-WPO unfolding 版** | **~4.0M** |
| **8** | **WaveMST_KG_Unfold** | **KG unfolding 版** | **~4.0M** |

> Model 0/1 保留用于对照实验。参数量：share_weights=True 时约 0.85M，False 时约 0.79M × K。

---

## 4. 配置项说明

在 `train.py` 顶部的 CONFIG 区域修改：

### 4.1 基础配置（与 Stage 1 相同）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_INDEX` | 7 | 选择模型，7=Unfold_3D，8=Unfold_KG |
| `GPU_ID` | '0' | GPU 编号 |
| `BATCH_SIZE` | 3 | 训练 batch 大小 |
| `MAX_EPOCH` | 300 | 总训练 epoch 数 |
| `LR` | 4e-4 | 学习率 |
| `CROP_SIZE` | 256 | 训练裁剪尺寸 |
| `MASK_MODE` | 'A' | Mask 方案：A / B / D |

### 4.2 Unfolding 专用配置（仅 MODEL_INDEX >= 7 时生效）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `NUM_STAGES` | 5 | Unfolding 迭代次数 K，推荐 3/5/7/9 |
| `SHARE_STAGE_WEIGHTS` | False | True: 所有 stage 共享 WPO 权重（参数量不变）<br>False: 每 stage 独立权重（参数量 × K） |
| `MULTI_STAGE_LOSS` | True | True: 对最后 4 个 stage 加权求损失（DPU 风格）<br>False: 仅用最后 1 个 stage 的损失 |

### 4.3 数据路径

数据集位于上级目录 `../dataset/`，与 Stage 1 共用，**无需额外操作**：
```
../dataset/
├── CAVE_1024_npy/          # 训练集
├── TSA_simu_data/
│   ├── Truth/              # 测试集
│   └── mask.mat            # CASSI mask
```

---

## 5. 快速上手

### 5.1 首次训练（推荐配置）

打开 `train.py`，确认以下配置：

```python
MODEL_INDEX         = 7       # 3D-WPO Unfolding
BATCH_SIZE          = 3       # 5stg + 独立权重需要降 batch
NUM_STAGES          = 5
SHARE_STAGE_WEIGHTS = False
MULTI_STAGE_LOSS    = True
```

运行：

```bash
cd G:\MachineLearning\CASSI\stage2
python train.py
```

### 5.2 训练进度参考

正常训练应当看到以下趋势：

| Epoch | 预期 PSNR | 预期 Loss |
|:-----:|:---------:|:---------:|
| 1 | ~22–24 | ~0.09 |
| 30 | ~33–34 | ~0.023 |
| 100 | ~35.5–36 | ~0.017 |
| 200 | ~37–37.5 | ~0.015 |
| 300 | ~37.5–38 | ~0.014 |

> 如果 30 epoch 后 PSNR 仍低于 30，说明实现可能有问题，请检查。

### 5.3 测试

修改 `test.py` 中的 `CHECKPOINT` 路径，确保 `MODEL_INDEX` 和 `NUM_STAGES`、`SHARE_STAGE_WEIGHTS` 与训练时一致：

```python
MODEL_INDEX         = 7
CHECKPOINT          = 'result/model/2026_xx_xx_xx_xx_xx_3d_wpo_unfold_stg5/best.pth'
NUM_STAGES          = 5
SHARE_STAGE_WEIGHTS = False
```

运行：

```bash
python test.py
```

测试输出会额外打印 **Stage-wise PSNR**，展示每个 unfolding stage 的重建质量演化：

```
  Stage-wise PSNR:
    Stage 1: PSNR=28.32
    Stage 2: PSNR=32.15
    Stage 3: PSNR=34.87
    Stage 4: PSNR=36.52
    Stage 5: PSNR=37.41
```

---

## 6. 推荐实验矩阵

按优先级排列：

| 编号 | MODEL | STAGES | SHARE | BATCH | 预期 PSNR | 说明 |
|:----:|:-----:|:------:|:-----:|:-----:|:---------:|------|
| Run-1 | 7 | 5 | False | 3 | ~37.5 | **首选，验证框架** |
| Run-2 | 7 | 5 | True | 5 | ~36.5 | 共享权重基线 |
| Run-3 | 7 | 9 | False | 2 | ~38.5 | 极限性能 |
| Run-4 | 8 | 5 | False | 3 | ~37.5 | KG 版本 |
| Run-5 | 8 | 9 | True | 5 | ~37.8 | KG + 共享 + 长迭代 |

> **优先跑 Run-1**。如果 PSNR 在 30 epoch 后超过 33，说明框架正确，再扩展其他实验。

---

## 7. BATCH_SIZE 与显存

Unfolding 模型每个 stage 都保存中间激活用于反向传播，显存占用远大于单 stage 模型。

| 配置 | 建议 BATCH_SIZE | 预估显存（24GB GPU） |
|------|:--------------:|:-------------------:|
| 3stg, share=False | 5 | ~14 GB |
| 5stg, share=False | 3 | ~16 GB |
| 5stg, share=True | 5 | ~12 GB |
| 9stg, share=False | 2 | ~20 GB |
| 9stg, share=True | 5 | ~16 GB |

> 如果 OOM，优先降 BATCH_SIZE，其次改 share_weights=True。

---

## 8. Unfolding 数学原理简述

### 8.1 GAP (Generalized Alternating Projection) 框架

CASSI 测量模型：$g = \Phi f + n$

每个 stage 的迭代公式：

$$z = f + \rho_k \cdot \Phi^T \frac{g - \Phi f}{\Phi \Phi^T}$$

$$f^{(k+1)} = \text{WPO3D}(z, \text{mask})$$

其中：
- $\Phi f$：shift(mask * f)，再沿光谱维 sum → 模拟 CASSI 测量
- $\Phi^T r$：将残差广播到各波段，乘 mask，再 shift_back → 反投影
- $\Phi \Phi^T$：预计算的对角矩阵（每像素 mask 平方和），防止除零
- $\rho_k$：可学习步长，由 ParaEstimator 从当前 $f$ 预测

### 8.2 Multi-stage Loss

仅用最后 stage 的 loss 训练不稳定。DPU 风格的加权 loss：

$$\mathcal{L} = \text{RMSE}(f^K, \text{GT}) + 0.7 \cdot \text{RMSE}(f^{K-1}, \text{GT}) + 0.5 \cdot \text{RMSE}(f^{K-2}, \text{GT}) + 0.3 \cdot \text{RMSE}(f^{K-3}, \text{GT})$$

对最后 4 个 stage 递减加权，确保中间 stage 也产生合理的中间结果。

### 8.3 Share Weights vs. Independent Weights

| 模式 | 参数量 | 特点 |
|------|:------:|------|
| share_weights=True | 0.79M + 少量 rho | 所有 stage 共用同一个 WPO，靠迭代次数提升 |
| share_weights=False | 0.79M × K | 每 stage 独立 WPO，表达力更强但参数更多 |

> ParaEstimator（步长预测器）始终每 stage 独立，不共享。

---

## 9. Checkpoint 命名规则

训练产生的 checkpoint 存放在 `result/model/` 下，目录名自动包含时间戳和模型标签：

```
result/model/
└── 2026_05_08_15_30_00_3d_wpo_unfold_stg5/
    ├── best.pth          # 最优 PSNR 的 checkpoint
    ├── epoch_050.pth     # 每 50 epoch 保存
    ├── epoch_100.pth
    └── ...
```

- `_stg5` 表示 5 stage unfolding
- `_share` 后缀表示 share_weights=True
- `_kg` 表示 KG 版本

---

## 10. 常见问题

### Q: 训练初期 loss 很大或者 NaN？

GD step 的残差在训练初期可能很大。代码中已有 `residual.clamp(min=-10, max=10)` 和 `PhiPhiT.clamp(min=1e-6)` 保护。如果仍有问题，尝试降低学习率到 `2e-4`。

### Q: share_weights=True 时训练发散？

共享权重时梯度会从 K 个 stage 累积。尝试降低学习率到 `LR / sqrt(K)`，例如 K=5 时用 `1.8e-4`。

### Q: 可以加载 Stage 1 的 checkpoint 吗？

不能直接加载。Unfolding 模型的 state_dict 包含 `rho_estimators` 和 `initial_conv` 等新参数，与 Stage 1 不兼容。但可以手动加载 prior 部分的权重做预训练初始化（高级用法，暂不提供）。

### Q: 为什么 Model 0/1 还在代码里？

保留用于对照实验。设置 `MODEL_INDEX=0` 即可退回 Stage 1 的非 unfolding 模式，训练流程不变。

### Q: test.py 的 Stage-wise PSNR 不单调上升？

正常情况应单调上升。如果中间 stage 出现回退，可能是 MULTI_STAGE_LOSS 没开（中间 stage 没有训练信号）。确保训练时 `MULTI_STAGE_LOSS = True`。

---

## 11. 与 Stage 1 的关系

| 项目 | Stage 1 | Stage 2 |
|------|---------|---------|
| 目录 | `G:\MachineLearning\CASSI\` | `G:\MachineLearning\CASSI\stage2\` |
| 模型 | Model 0–6（end-to-end） | Model 7–8（unfolding） |
| WPO 模块 | `wpo3d.py`（原始） | `wpo3d.py`（复制，不修改） |
| 数据集 | `./dataset/` | `../dataset/`（共用） |
| Mask 机制 | A / B / D | A / B / D（透传给内部 WPO） |
| 训练方式 | 单次 forward | K 次 GD + Prior 迭代 |
| 输入格式 | shift_back(g)（H setting） | 原始测量 g + PhiPhiT |
