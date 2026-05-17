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

### 1.1 数据集元信息 JSON（自动生成）

**首次** 运行 `python train.py` 时，`load_training()` 会边加载数据边扫描，在训练集目录下自动生成两个 JSON 文件：

```
dataset/CAVE_1024_npy/
├── CAVE_1024_npy_info.json    ← 场景列表、band 数、波长范围
└── CAVE_1024_npy_bands.json   ← 每个 band 的全局均值 + 波长（nm）
```

`bands.json` 示例：
```json
{
  "data_name": "CAVE_1024_npy",
  "num_bands": 28,
  "wavelengths_nm": [453, 457, 462, ...],
  "band_mean": [0.3121, 0.3184, ...]
}
```

**两个 JSON 均已存在时，后续启动直接跳过扫描**，不做重复工作。

这两个 JSON 还用于：
- 训练参数 `NUM_BANDS` 自动从 `bands.json` 读取，不再依赖 CONFIG 硬编码
- `physics.py` 的 `WAVELENGTHS` 自动从 `bands.json` 读取，换数据集后自动更新

---

## 2. 训练

### 2.1 修改 CONFIG

打开 `train.py`，找到顶部 `CONFIG` 区域：

```python
MODEL_INDEX   = 0       # ← 选择模型（见下表）
BATCH_SIZE    = 5       # ← 根据显存调整，24GB 可用 5
MAX_EPOCH     = 300
LR            = 4e-4
MASK_MODE     = 'A'     # ← mask 机制：'A'(默认)/'B'/'D'（各模型支持情况见 §5.3）
INPUT_SETTING = 'H'     # ← 'H' / 'HM' / 'Y'
TRAIN_PATH    = ...     # ← 自动检测 npy 或 mat，通常不用改
```

### 2.2 模型索引一览

| MODEL_INDEX | 模型 | 架构简述 | mask_mode 支持 |
|-------------|------|----------|----------------|
| 0 | WaveMST_3D | 纯 3D WPO，主推，速度最快 | A / B / D |
| 1 | WaveMST_KG | 3D WPO + Klein-Gordon Born 修正（固定 D） | D（固定） |
| 2 | WaveMST_Parallel | WPO 并联 S-MSA | A / B / D |
| 3 | WaveMST_Mamba | 2D WPO + 1D Mamba，需 mamba-ssm | A / B / D |
| 4 | WaveMST_Phys | H2-α：物理波数注入 WPO（2D 空间 FFT） | A / B / D |
| 5 | Helmholtzformer | H1-γ：纯稳态亥姆霍兹逆算子，消融基准 | **A / B 仅** |
| 6 | WaveMST_Helm | **H2-γ：三合一主推**（WPO + Beer-Lambert 吸收） | A / B / D |

> **注意**：Model 1 (`WaveMST_KG`) 内部强制使用 Klein-Gordon Born 修正（等价于 mask_mode='D'），
> CONFIG 中的 `MASK_MODE` 对它无效。
>
> **注意**：Model 5 (`Helmholtzformer`) 不支持 `MASK_MODE='D'`，
> 因为 Klein-Gordon Born 修正依赖波动方程中间量，与静态亥姆霍兹算子不兼容。
> 设置 `MASK_MODE='D'` 后运行 Model 5 会在构建时抛出 `ValueError`，这是预期行为。

### 2.3 启动训练

```bash
python train.py
```

输出格式（更新后）：
```
训练集路径: dataset/CAVE_1024_npy
starting at: Sun May  4 02:19:00 2026
[dataset] JSON 已存在，跳过扫描: CAVE_1024_npy_info.json / CAVE_1024_npy_bands.json
训练集加载完成：205 个场景
[train] NUM_BANDS=28（来自 CAVE_1024_npy_bands.json）
[physics] WAVELENGTHS 已从 CAVE_1024_npy_bands.json 加载，共 28 个波段。
模型: WaveMST_Phys  参数量: 6.83M
训练开始: 5.4.2:19
[Epoch 001] Loss: 0.042314  Time: 48.3s  LR: 4.00e-04
             Test → PSNR: 27.83  SSIM: 0.8721  SAM: 0.0412
                    FreqAmpErr: 0.00821  LowFreqErr: 0.00234  HighFreqErr: 0.01203
  ★ 新最优: PSNR=27.83  SSIM=0.8721  → result/model/.../best.pth
...
[Epoch 050] Loss: 0.018231  Time: 46.1s  LR: 3.89e-04
[Epoch 050] 当前时间: 5.4.3:25  预计结束: 5.4.8:42
```

训练指标说明：

| 指标 | 含义 |
|------|------|
| PSNR | 峰值信噪比（越高越好，目标 > 32 dB） |
| SSIM | 结构相似度（越高越好，目标 > 0.92） |
| SAM | 光谱角（越低越好，弧度，目标 < 0.04） |
| FreqAmpErr | 频域幅度谱 MSE（越低越好，体现频域重建质量） |
| LowFreqErr | 低频段（r < 10% 截止半径）幅度差异 |
| HighFreqErr | 高频段幅度差异（纹理细节） |

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

`show_all` 输出的图像（每个场景一个子目录）：

| 文件 | 内容 |
|------|------|
| `pred_bands.png` / `gt_bands.png` | 多波段灰度图 |
| `pred_rgb.png` / `gt_rgb.png` | 伪彩色 RGB 合成 |
| `spectra.png` | 5 个空间位置的光谱曲线对比 |
| `compare_band{N}.png` | 中间波段：GT / Pred / 误差 三联图 |
| `error_map.png` | 全波段平均绝对误差热图 |
| `pred_freq.png` / `gt_freq.png` | **频域幅度谱（log 尺度，DC 居中）** |
| `freq_compare_band{N}.png` | **中间波段频域并排对比：GT / Pred / 差值** |
| `rapsd.png` | **径向平均功率谱密度曲线对比（log 纵轴）** |

频域图组用于体现模型在频域的重建特性，直接辅助分析低频失真和高频细节损失。

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

### 5.3 Mask 机制选择

#### 三种方案的原理

| MASK_MODE | 机制 | 适用模型 | 推荐用途 |
|-----------|------|----------|----------|
| `'A'` | **初始振幅软门控**：`u0 = φ(x)·gate, v0 = ψ(x)·gate`，`gate = 0.1 + 0.9·mask` | 全部（0~6） | 首选，训练最稳定 |
| `'B'` | **Mask 作为源项**：WPO 零阶传播后叠加 mask 调制的源贡献 | 0,2,3,4,5,6 | 消融实验，验证源项注入的贡献 |
| `'D'` | **Klein-Gordon Born 修正**：质量场 `m²=(1-mask)·m0²`，一阶 Born 修正 | 0,2,3,4,6（**不含 5**） | 效果验证后切换，Model 1 固定使用 D |

#### 各模型 mask_mode 支持矩阵

| 模型 | A | B | D | 备注 |
|------|:-:|:-:|:-:|------|
| 0 WaveMST_3D | ✅ | ✅ | ✅ | 3D FFT，mask 控制初始条件 |
| 1 WaveMST_KG | — | — | 固定 | CONFIG 中 MASK_MODE 对此模型无效，内部强制 D |
| 2 WaveMST_Parallel | ✅ | ✅ | ✅ | WPO 分支用 mask_mode，S-MSA 分支不受影响 |
| 3 WaveMST_Mamba | ✅ | ✅ | ✅ | 同 Model 0 |
| 4 WaveMST_Phys | ✅ | ✅ | ✅ | 2D FFT，修正也用 2D FFT（与模型一致） |
| 5 Helmholtzformer | ✅ | ✅ | ❌ | D 不支持（见下方说明） |
| 6 WaveMST_Helm | ✅ | ✅ | ✅ | mask_mode 控制 WPO 部分；Beer-Lambert 吸收不受 mask_mode 影响 |

#### 为什么 Model 5 不支持 mask_mode='D'？

`MaskKleinGordonD` 的 Born 修正需要波动方程传播的中间量（sinc 项和 decay）来做频域修正。
亥姆霍兹逆算子是静态算子（无时间演化），不产生这些中间量，
因此在 `Helmholtzformer` 上使用 `'D'` 在物理上没有意义，代码层面也会在构建时抛出 `ValueError`：

```
ValueError: Helmholtzformer 不支持 mask_mode='D'（Klein-Gordon Born 修正依赖波动方程
中间量，与静态亥姆霍兹算子不兼容）。请使用 'A' 或 'B'。
```

#### 实验策略建议

```
第一步：MASK_MODE='A' 跑通所有模型，确认收敛
第二步：切换 MASK_MODE='B'，对比 A vs B（源项 vs 门控）
第三步：切换 MASK_MODE='D'（Model 0/4/6），验证 Born 修正的提升
        ※ Model 5 保持 A 或 B 参与对比
```

### 5.4 Model 4 vs Model 6 的差异

| 对比点 | Model 4 (WaveMST_Phys) | Model 6 (WaveMST_Helm) |
|--------|------------------------|------------------------|
| 传播算子 | 2D WPO + 物理波数 | 2D WPO + 物理波数（同 4） |
| 额外步骤 | 无 | + Beer-Lambert 吸收（Step 3） |
| 参数量 | 较少 | 略多（每个 Block 多 κ₀、L 参数） |
| mask 作用 | 控制初始条件（Step 1） | 控制初始条件 + 控制吸收率（Step 3） |
| 预期效果 | 物理波数先验的基线 | 三合一，预期最强 |

Model 6 的 Beer-Lambert 吸收始终使用 mask 计算透射率，**不受 MASK_MODE 影响**——这是符合物理的设计：
吸收衰减由 mask 的遮挡程度决定，与初始条件的生成方式无关。

### 5.5 INPUT_SETTING 选择

| 设置 | 输入到模型 | 特点 |
|------|-----------|------|
| `'H'`  | `shift_back(meas / 28 * 2)` | 最常用，MST 默认 |
| `'HM'` | `H * mask3d` | 额外乘 mask，信息更密集 |
| `'Y'`  | 原始测量值（未展开）| 需要适配模型输入维度 |

### 5.6 训练 epoch 建议

| 阶段 | epoch | 目的 |
|------|-------|------|
| 快速验证 | 50 | 看模型能否正常收敛，PSNR > 28 dB |
| 初步结果 | 150 | PSNR 趋于稳定 |
| 正式实验 | 300 | 论文最终结果 |

---

## 6. 消融实验流程

```python
# 推荐实验矩阵（在 train.py CONFIG 中切换）：

# WaveMST 系列
MODEL_INDEX=0, MASK_MODE='A'   # WaveMST_3D 基线
MODEL_INDEX=0, MASK_MODE='D'   # WaveMST_3D + KG Born 修正（= Model 1 的近亲）
MODEL_INDEX=4, MASK_MODE='A'   # H2-α：物理波数先验的贡献
MODEL_INDEX=6, MASK_MODE='A'   # H2-γ：三合一主推

# Helmholtz 消融
MODEL_INDEX=5, MASK_MODE='A'   # H1-γ：去掉动态传播，看稳态算子的贡献
MODEL_INDEX=5, MASK_MODE='B'   # H1-γ + 源项调制

# Mask 机制消融（以 Model 0 为例，控制变量）
MODEL_INDEX=0, MASK_MODE='A'   # 软门控
MODEL_INDEX=0, MASK_MODE='B'   # 源项注入
MODEL_INDEX=0, MASK_MODE='D'   # Born 修正（= Model 1）
```

### 6.1 调试新模型的检查清单

每实现/首次运行一个新模型，依次确认：

```bash
# 1. 实例化无报错
python -c "from wpo3d_phys import WaveMST_Phys; m=WaveMST_Phys(mask_mode='A'); print('OK')"
python -c "from helm_pure import Helmholtzformer; m=Helmholtzformer(mask_mode='B'); print('OK')"

# 2. 输出形状正确 [B, 28, 256, 256]
# 3. Backward 无 NaN：loss.backward() 后检查梯度
# 4. 10 epoch 后 Loss 下降，PSNR > 25 dB
# 5. 50 epoch 后 PSNR > 30 dB
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
| `NaN loss`（Helmholtz 模型） | 频域分母太小（ε 不够大） | 调大 `eps_raw` 初始值，或检查 `denom.abs().min()` |
| Model 4/6 性能接近 Model 0 | 物理波数尺度与空间频率不匹配 | 打印 `k_phys.min(), k_phys.max()`，确认在 [0.66, 1.0] |
| `RuntimeError: Expected complex` | PyTorch 版本不支持复数 tensor | 需要 PyTorch ≥ 1.7 |
| `ValueError: Helmholtzformer 不支持 mask_mode='D'` | Model 5 + MASK_MODE='D' | 改为 `MASK_MODE='A'` 或 `'B'`，或换用 Model 0/4/6 |
| `bands.json` 中 `wavelengths_nm` 为 null | 首次加载时未传 wavelengths 参数 | 删除旧 json，重新运行 `python train.py`，json 会自动重建 |
