# WaveMST 使用手册

> 快速上手：如何运行、调参、解读结果。

---

## 0. 环境准备

```bash
pip install torch torchvision scipy numpy matplotlib einops
# 可选（Model 3 光谱 SSM 加速）：
pip install mamba-ssm
# 可选（FLOPs 统计）：
pip install thop
```

---

## 1. 数据准备（只做一次）

```bash
# 把所有 .mat 文件转换为 .npy（约需 10~30 分钟）
python dataset/mat2npy.py
```

转换后目录结构：
```
dataset/
├── CAVE_1024_npy/    ← scene1.npy ~ scene205.npy  (训练集)
├── CAVE_512_npy/     ← scene01.npy ~ scene30.npy  (可选)
└── TSA_simu_data/
    ├── Truth/        ← scene01.npy ~ scene10.npy  (测试集)
    └── mask.npy      ← 已存在
```

---

## 2. 训练

### 2.1 修改 CONFIG

打开 `train.py`，找到顶部 `CONFIG` 区域：

```python
MODEL_INDEX  = 0       # ← 选择模型（见下表）
BATCH_SIZE   = 5       # ← 根据显存调整，24GB 可用 5
MAX_EPOCH    = 300
LR           = 4e-4
MASK_MODE    = 'A'     # ← mask 机制：'A'(默认)/'B'/'D'
TRAIN_PATH   = ...     # ← 自动检测 npy 或 mat，通常不用改
```

| MODEL_INDEX | 模型 | 说明 |
|-------------|------|------|
| 0 | WaveMST_3D | 纯 3D WPO，主推，速度最快 |
| 1 | WaveMST_KG | 3D WPO + Klein-Gordon Born 修正 |
| 2 | WaveMST_Parallel | WPO 并联 S-MSA，最稳妥 |
| 3 | WaveMST_Mamba | 2D WPO + 1D Mamba，需 mamba-ssm |

### 2.2 启动训练

```bash
python train.py
```

输出格式：
```
[Epoch 001] Loss: 0.042314  Time: 48.3s  LR: 4.00e-04
             Test → PSNR: 27.83  SSIM: 0.8721
  ★ 新最优: PSNR=27.83 SSIM=0.8721 → result/model/2026_04_23_10_00_00_3d_wpo_pure/best.pth
```

checkpoint 保存在 `result/model/<时间戳>_<模型名>/`。

---

## 3. 测试

```python
# 修改 test.py 的 CONFIG：
MODEL_INDEX = 0
CHECKPOINT  = 'result/model/2026_04_23_10_00_00_3d_wpo_pure/best.pth'
```

```bash
python test.py
```

输出：
```
  Scene 01: PSNR=33.21  SSIM=0.9412  SAM=0.0312
  ...
  平均:    PSNR=32.87  SSIM=0.9381  SAM=0.0328
结果保存到: result/show/2026_04_23_test_3d_wpo_pure/
```

---

## 4. 可视化

```python
from viz import show_all
import numpy as np

pred = np.load('result/show/.../pred.npy')  # [10, 28, 256, 256]
gt   = np.load('result/show/.../gt.npy')

show_all(pred, gt, scene_idx=0, save_dir='result/show/viz')
```

或在测试后直接调用（test.py 结尾加几行即可）。

---

## 5. 关键调参指南

### 5.1 显存不足

```python
BATCH_SIZE = 2      # 降低 batch size
STAGE = 1           # 减少 U-Net 层数（损失一些性能）
NUM_BLOCKS = [1,1,1]
DIM = 28            # 不要改小（28 是 CASSI 波段数，有物理含义）
```

### 5.2 训练不收敛

```python
LR = 1e-4           # 降低学习率（默认 4e-4 有时偏大）
SCHEDULER = 'MultiStepLR'  # 换成阶梯调度
MILESTONES = [100, 200, 250]
```

### 5.3 选择 mask 机制

| MASK_MODE | 特点 | 推荐场景 |
|-----------|------|----------|
| 'A' | 稳定，训练容易 | 首选，baseline |
| 'B' | 源项注入，稍慢 | 消融实验 |
| 'D' | Klein-Gordon Born 修正，创新最强 | 效果验证后切换 |

先用 `'A'` 跑通，确认性能后再尝试 `'D'`。

### 5.4 INPUT_SETTING 选择

| 设置 | 输入到模型 | 特点 |
|------|-----------|------|
| 'H'  | shift_back(meas/28*2) | 最常用，MST 默认 |
| 'HM' | H * mask3d | 额外乘 mask，信息更密集 |
| 'Y'  | 原始测量值（未展开）| 需要适配模型输入维度 |

### 5.5 训练 epoch 建议

| 阶段 | epoch | 目的 |
|------|-------|------|
| 快速验证 | 50 | 看模型能否正常收敛 |
| 初步结果 | 150 | PSNR 趋于稳定 |
| 正式实验 | 300 | 论文最终结果 |

---

## 6. 消融实验流程

```python
# 依次修改 MODEL_INDEX 和 MASK_MODE，各跑一次 train.py
# 建议实验矩阵：

实验1: MODEL_INDEX=0, MASK_MODE='A'   ← 主模型
实验2: MODEL_INDEX=2, MASK_MODE='A'   ← + S-MSA (Model 2)
实验3: MODEL_INDEX=0, MASK_MODE='D'   ← + KG Born (Model 1)
MST baseline: python MST/simulation/train_code/train.py
```

---

## 7. 常见报错

| 报错 | 原因 | 解决 |
|------|------|------|
| `FileNotFoundError: mask.npy` | 数据未转换 | 先跑 `python dataset/mat2npy.py` |
| `CUDA out of memory` | 显存不足 | 减小 `BATCH_SIZE` 或 `STAGE` |
| `NaN loss` | WPO 参数溢出 | 减小 `LR`；检查 `softplus` 是否生效 |
| `ModuleNotFoundError: mamba_ssm` | Mamba 未安装 | `pip install mamba-ssm` 或改用 Model 0/1/2 |
| `einops` 报错 | einops 未安装 | `pip install einops` |

# WaveMST new使用手册

> 快速上手：如何运行、调参、解读结果。

---

## 0. 环境准备

```bash
pip install torch torchvision scipy numpy matplotlib einops
# 可选（Model 3 光谱 SSM 加速）：
pip install mamba-ssm
# 可选（FLOPs 统计）：
pip install thop
```

---

## 1. 数据准备（只做一次）

```bash
# 把所有 .mat 文件转换为 .npy（约需 10~30 分钟）
python dataset/mat2npy.py
```

转换后目录结构：
```
dataset/
├── CAVE_1024_npy/    ← scene1.npy ~ scene205.npy  (训练集)
├── CAVE_512_npy/     ← scene01.npy ~ scene30.npy  (可选)
└── TSA_simu_data/
    ├── Truth/        ← scene01.npy ~ scene10.npy  (测试集)
    └── mask.npy      ← 已存在
```

---

## 2. 训练

### 2.1 修改 CONFIG

打开 `train.py`，找到顶部 `CONFIG` 区域：

```python
MODEL_INDEX  = 0       # ← 选择模型（见下表）
BATCH_SIZE   = 5       # ← 根据显存调整，24GB 可用 5
MAX_EPOCH    = 300
LR           = 4e-4
MASK_MODE    = 'A'     # ← mask 机制：'A'(默认)/'B'/'D'
TRAIN_PATH   = ...     # ← 自动检测 npy 或 mat，通常不用改
```

| MODEL_INDEX | 模型 | 说明 |
|-------------|------|------|
| 0 | WaveMST_3D | 纯 3D WPO，主推，速度最快 |
| 1 | WaveMST_KG | 3D WPO + Klein-Gordon Born 修正 |
| 2 | WaveMST_Parallel | WPO 并联 S-MSA，最稳妥 |
| 3 | WaveMST_Mamba | 2D WPO + 1D Mamba，需 mamba-ssm |
| 4 | WaveMST_Phys | H2-α：物理波数注入 WPO（2D FFT），Model 0 的近亲 |
| 5 | Helmholtzformer | H1-γ：纯稳态亥姆霍兹逆算子，消融基准 |
| 6 | WaveMST_Helm | **H2-γ：三合一主推方案**（WPO + Beer-Lambert 吸收） |

### 2.2 启动训练

```bash
python train.py
```

输出格式：
```
[Epoch 001] Loss: 0.042314  Time: 48.3s  LR: 4.00e-04
             Test → PSNR: 27.83  SSIM: 0.8721
  ★ 新最优: PSNR=27.83 SSIM=0.8721 → result/model/2026_04_23_10_00_00_3d_wpo_pure/best.pth
```

checkpoint 保存在 `result/model/<时间戳>_<模型名>/`。

---

## 3. 测试

```python
# 修改 test.py 的 CONFIG：
MODEL_INDEX = 0
CHECKPOINT  = 'result/model/2026_04_23_10_00_00_3d_wpo_pure/best.pth'
```

```bash
python test.py
```

输出：
```
  Scene 01: PSNR=33.21  SSIM=0.9412  SAM=0.0312
  ...
  平均:    PSNR=32.87  SSIM=0.9381  SAM=0.0328
结果保存到: result/show/2026_04_23_test_3d_wpo_pure/
```

---

## 4. 可视化

```python
from viz import show_all
import numpy as np

pred = np.load('result/show/.../pred.npy')  # [10, 28, 256, 256]
gt   = np.load('result/show/.../gt.npy')

show_all(pred, gt, scene_idx=0, save_dir='result/show/viz')
```

或在测试后直接调用（test.py 结尾加几行即可）。

---

## 5. 关键调参指南

### 5.1 显存不足

```python
BATCH_SIZE = 2      # 降低 batch size
STAGE = 1           # 减少 U-Net 层数（损失一些性能）
NUM_BLOCKS = [1,1,1]
DIM = 28            # 不要改小（28 是 CASSI 波段数，有物理含义）
```

### 5.2 训练不收敛

```python
LR = 1e-4           # 降低学习率（默认 4e-4 有时偏大）
SCHEDULER = 'MultiStepLR'  # 换成阶梯调度
MILESTONES = [100, 200, 250]
```

### 5.3 选择 mask 机制

| MASK_MODE | 特点 | 推荐场景 |
|-----------|------|----------|
| 'A' | 稳定，训练容易 | 首选，baseline |
| 'B' | 源项注入，稍慢 | 消融实验 |
| 'D' | Klein-Gordon Born 修正，创新最强 | 效果验证后切换 |

先用 `'A'` 跑通，确认性能后再尝试 `'D'`。

### 5.4 INPUT_SETTING 选择

| 设置 | 输入到模型 | 特点 |
|------|-----------|------|
| 'H'  | shift_back(meas/28*2) | 最常用，MST 默认 |
| 'HM' | H * mask3d | 额外乘 mask，信息更密集 |
| 'Y'  | 原始测量值（未展开）| 需要适配模型输入维度 |

### 5.5 训练 epoch 建议

| 阶段 | epoch | 目的 |
|------|-------|------|
| 快速验证 | 50 | 看模型能否正常收敛 |
| 初步结果 | 150 | PSNR 趋于稳定 |
| 正式实验 | 300 | 论文最终结果 |

---

## 6. 消融实验流程

```python
# WaveMST 系列 vs Helmholtz 系列
# 建议实验矩阵：

Baseline:    MODEL_INDEX=0, MASK_MODE='A'   # WaveMST_3D（3D WPO）
H2-α:        MODEL_INDEX=4                  # 物理波数注入 WPO（2D FFT）
H1-γ:        MODEL_INDEX=5                  # 纯稳态亥姆霍兹（消融：去动态传播）
H2-γ（主推）: MODEL_INDEX=6                  # 三合一：WPO + Beer-Lambert
MST baseline: python MST/simulation/train_code/train.py

# Helmholtz 系列的消融（通过修改代码实现）：
# -Step3: 去掉 BeerLambertAbsorption → 等价于 Model 4
# -k_phys: 随机初始化 k_learn（测试物理先验贡献）
# -Step1M: 去掉初始门控的 mask → gate = 1
```

### 6.1 调试新模型的检查清单

每实现/首次运行一个新模型，依次确认：

1. 实例化无报错：`python -c "from wpo3d_phys import WaveMST_Phys; m=WaveMST_Phys()"`
2. Forward 输出形状正确：`[B, 28, 256, 256]`
3. Backward 无 NaN：`loss.backward()` 后检查梯度
4. 10 epoch 后 Loss 下降，PSNR > 25 dB
5. 50 epoch 后 PSNR > 30 dB（低于此值检查 §9 陷阱列表）

---

## 7. 常见报错

| 报错 | 原因 | 解决 |
|------|------|------|
| `FileNotFoundError: mask.npy` | 数据未转换 | 先跑 `python dataset/mat2npy.py` |
| `CUDA out of memory` | 显存不足 | 减小 `BATCH_SIZE` 或 `STAGE` |
| `NaN loss` | WPO 参数溢出 | 减小 `LR`；检查 `softplus` 是否生效 |
| `ModuleNotFoundError: mamba_ssm` | Mamba 未安装 | `pip install mamba-ssm` 或改用 Model 0/1/2 |
| `einops` 报错 | einops 未安装 | `pip install einops` |
| `NaN loss`（Helmholtz 模型） | 频域分母太小（ε 不够大） | 调大 `eps_raw` 初始值，或检查 `denom.abs().min()` |
| Model 4/6 性能接近 Model 0 | 物理波数尺度与空间频率不匹配 | 打印 `k_phys.min(), k_phys.max()`，确认在 [0.66, 1.0] |
| `RuntimeError: Expected complex`（Helmholtz） | PyTorch 版本不支持复数 tensor | 需要 PyTorch ≥ 1.7，当前环境 1.12 应支持 |
