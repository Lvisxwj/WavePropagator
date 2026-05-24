# Version2 快速上手 & 验证指南

## 文件结构

```
version2/
├── config.yaml          # 超参数（GPU, batch, LR, 数据路径）
├── __init__.py          # 模型开关（FBGW, Swin, KG, stages）
├── train.py             # 训练入口
├── test.py              # 测试入口
├── dataset.py           # 数据加载（路径从 config.yaml 读取）
└── model/
    ├── wpo3d.py         # WPO3D 核心 + U-Net（+FBGW +Swin +sigma）
    ├── unfolding.py     # A-HQS unfolding（+动量 +退化估计 +精化）
    ├── degradation.py   # 三合一退化估计（NEW）
    ├── refinement.py    # 局部精化 DWConv FFN（NEW）
    ├── mask_ops.py      # MaskGateA / MaskKleinGordonD
    └── utils.py         # shift/shift_back/ParaEstimator
```

## 架构概览

**Purify → Propagate → Refine 范式**

每个 unfolding stage 的 7 步流水线：

```
1. 退化估计 → ΔΦ, deg_weight, σ
2. Nesterov 动量外推
3. 修正 GD step（Φ_eff = Φ + ΔΦ）
4. 初始场净化（z * deg_weight + z）
5. WPO 传播（σ 控制阻尼，内含全局残差 z_clean）
6. 局部精化（DWConv FFN）
7. 输出：f = f_wave + f_local
```

**对比 Version1（GAP unfolding）：**
- GAP → A-HQS（二阶 Nesterov 动量）
- 新增退化估计（~5.2K 参数，修正 sensing error + 噪声感知）
- 新增局部精化（~4.7K 参数，补充纹理细节）
- 可选 FBGW 频带加权（零参数 SNR 自适应 / 可学习频带）
- 可选 Swin-WPO（64×64 窗内传播，降低 FFT 复杂度）

-----

## Step 0: 配置

### config.yaml

```yaml
gpu_id: '0'
batch_size: 8           # 根据显存调（24G→8, 16G→5）
max_epoch: 300
learning_rate: 4.0e-4
scheduler: 'CosineAnnealingLR'
crop_size: 256
dim: 28
unet_stage: 3
num_blocks: [2, 2, 2]

# 数据路径（绝对路径，按实际修改）
data_root: '/data5/SCI/xieweijie/CASSI/dataset'
train_path: '/data5/SCI/xieweijie/CASSI/dataset/CAVE_1024_npy'
test_path: '/data5/SCI/xieweijie/CASSI/dataset/TSA_simu_data/Truth'
mask_path: '/data5/SCI/xieweijie/CASSI/dataset/TSA_simu_data'
```

### __init__.py 开关说明

```python
USE_KG = False              # True → KG 方程（mask_mode='D'）
WPO_FBGW_MODE = 'none'     # 'none' / 'snr_adaptive' / 'learnable_band'
USE_SWIN_WPO = False        # True → 64×64 窗内传播
SWIN_WINDOW_SIZE = 64       # 窗大小（≥56）
USE_UNFOLDING = True        # True → deep unfolding
NUM_STAGES = 5              # unfolding stage 数
SHARE_STAGE_WEIGHTS = True  # True → 所有 stage 共享权重
MULTI_STAGE_LOSS = True     # True → 多 stage 加权损失
BEST_CKPT = ''              # test.py 用，填入 best.pth 路径
```

> 默认配置 = 纯 WPO + A-HQS + 退化估计 + 动量 + 精化，所有新模块已内置于 unfolding.py。

-----

## Step 1: 语法检查（本地，无需 GPU）

```bash
cd version2
python -c "
import ast
files = [
    'train.py', 'test.py', 'dataset.py',
    'model/wpo3d.py', 'model/unfolding.py',
    'model/degradation.py', 'model/refinement.py',
    'model/mask_ops.py', 'model/utils.py',
]
for f in files:
    ast.parse(open(f).read())
    print(f'  OK: {f}')
print('All syntax OK')
"
```

-----

## Step 2: 模型构建 + Forward 验证（服务器）

```bash
cd version2
python -c "
import sys, torch
sys.path.insert(0, '.')
from train import build_model, count_params, print_config

model = build_model().cuda()
print_config(model)

# 假数据 forward
B, C, H, W = 2, 28, 256, 256
g = torch.randn(B, 1, H, W + 54).cuda()       # W' = 256 + 27*2 = 310
Phi = torch.rand(B, C, H, W).cuda()
PhiPhiT = torch.ones(B, 1, H, W + 54).cuda()
outputs = model(g, (Phi, PhiPhiT))
print(f'Stages: {len(outputs)}, Output shape: {outputs[-1].shape}')
print('Forward pass OK!')
"
```

**预期输出：**
```
============================================================
当前配置组合:
  KG方程:     否
  FBGW:       none
  Swin-WPO:   否
  展开:       5 stage, 共享权重, A-HQS+动量
  多阶段损失: 是
  参数量:     X.XXM
============================================================
Stages: 5, Output shape: torch.Size([2, 28, 256, 256])
Forward pass OK!
```

-----

## Step 3: Baseline 跑 5 epoch（验证复制无误）

```bash
# __init__.py 保持默认
python train.py
```

**验收：** 前 5 epoch 的 loss 应正常下降，PSNR 应接近 version1/stage2 同 epoch 水平。

-----

## Step 4: 逐个开模块（每个跑 10 epoch）

**原则：每次只改一个开关，跑 10 epoch 看增量。无效就关掉。**

| 顺序 | 修改 __init__.py | 对比基准 | 关注指标 |
|------|-----------------|---------|---------|
| 4a | 默认（退化估计+动量+精化已内置） | version1 @10ep | PSNR 增量 |
| 4b | `WPO_FBGW_MODE = 'snr_adaptive'` | 4a @10ep | PSNR 增量 |
| 4c | `USE_SWIN_WPO = True` | 4a @10ep | PSNR + 速度 |
| 4d | `USE_KG = True` | 4a @10ep | SAM 变化 |

> 退化估计、Nesterov 动量、局部精化已集成在 unfolding.py，默认启用。
> Step 4a 本身就已经包含了所有 version2 新模块（vs version1 的纯 GAP）。

-----

## Step 5: 最佳组合跑满 300 epoch

选出 Step 4 中有效的模块组合：

```python
# 先跑 shared
SHARE_STAGE_WEIGHTS = True
# python train.py → 跑 300 epoch

# 再跑 non-shared
SHARE_STAGE_WEIGHTS = False
# python train.py → 跑 300 epoch
```

-----

## Step 6: 测试

```python
# __init__.py
BEST_CKPT = 'result/model/xxx/best.pth'  # 填入实际路径
```

```bash
python test.py
```

输出包含：每个 stage 的 PSNR 演化 + 每个 scene 的 PSNR/SSIM/SAM + 平均值。

-----

## 验收标准

| 配置 | PSNR 下限 | 说明 |
|------|----------|------|
| version1 5stg baseline | 38.21 | 已有结果 |
| version2 默认 @10ep | > baseline @10ep | 新模块有效 |
| version2 shared @300ep | **> 38.8** | 超 baseline 0.6+ dB |
| version2 non-shared @300ep | **> 39.2** | 逼近 DPU 水平 |

-----

## 常见问题

### OOM
- 降 `batch_size`（config.yaml）
- 关 `USE_SWIN_WPO`（窗切分增加中间变量）
- 开 `use_amp: true`（config.yaml）

### 训练发散
动量 beta 初始化过大 → 在 `model/unfolding.py` 中把：
```python
nn.Parameter(torch.tensor(0.0))    # sigmoid → 0.5
```
改成：
```python
nn.Parameter(torch.tensor(-2.0))   # sigmoid → 0.12，接近零动量
```

### Import 错误
确保从 `version2/` 目录运行：
```bash
cd /path/to/CASSI/version2
python train.py
```

### Swin-WPO 在深层退化为全局
这是正确行为：U-Net 下采样后 64×64 = 1 个窗 = 全局 WPO。深层应该全局建模。

-----

## 模块参数量参考

| 模块 | 新增参数 |
|------|---------|
| DegradationEstimation | ~5.2K |
| LocalRefinement | ~4.7K |
| Nesterov beta (×5) | 5 |
| lambda_sigma (per WPO) | 1 each |
| FBGW snr_adaptive | 0 |
| FBGW learnable_band | 8 |
| **总新增** | **< 10K** |

> WPO3D 核心闭式解完全不动，只在外围增加轻量增强。
